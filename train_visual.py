#!/usr/bin/env python3
"""Instrumented MLX LoRA training script with live parameter-space telemetry.

Uses mlx_lm.lora's exact training pipeline while recording:
- Train & validation loss curves
- Exact per-layer/per-module Frobenius parameter shift ||Delta W||_F and
  relative shift (%) compared to the frozen regular model ||W_0||_F
- Deterministic 3D random projections of the live parameter delta vectors
  (Full Model, Attention, MLP, and each Transformer layer L0..L35) relative to
  the frozen regular model at origin (0, 0, 0).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import yaml
from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.utils import linear_to_lora_layers


LAYER_RE = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.([a-zA-Z0-9_]+)$")


def _seeded_rademacher(key: str, shape: tuple[int, int]) -> mx.array:
    seed_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    seed_int = int.from_bytes(seed_bytes[:8], "little")
    rng = np.random.default_rng(seed_int)
    arr = rng.choice([-1.0, 1.0], size=shape).astype(np.float32)
    arr /= math.sqrt(shape[0])
    return mx.array(arr)


def _compute_base_weight_norm(linear: nn.Module) -> float:
    w = linear.weight
    if isinstance(linear, nn.QuantizedLinear):
        w = mx.dequantize(
            w,
            linear.scales,
            linear.biases,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
        )
    val = mx.sqrt(mx.sum(w.astype(mx.float32) ** 2))
    mx.eval(val)
    return float(val.item())


class ParameterSpaceCallback(TrainingCallback):
    """Logs loss curves and real-time LoRA parameter shifts vs base model."""

    def __init__(self, run_dir: Path, total_iters: int) -> None:
        self.run_dir = run_dir
        self.telemetry_path = run_dir / "telemetry.jsonl"
        self.meta_path = run_dir / "run_meta.json"
        self.total_iters = total_iters
        self.model: nn.Module | None = None
        self.modules: list[dict[str, Any]] = []
        self.prev_xyz: list[float] | None = None
        self.recorded_step0 = False

    def attach_model(
        self,
        model: nn.Module,
        config_dict: dict[str, Any],
        base_norms: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.modules = []
        for name, mod in model.named_modules():
            if not isinstance(mod, LoRALinear):
                continue
            m = LAYER_RE.search(name)
            layer_idx = int(m.group(1)) if m else -1
            proj_name = m.group(2) if m else name.split(".")[-1]
            group = "attn" if "self_attn" in name else ("mlp" if "mlp" in name else "other")
            out_dim, in_dim = mod.lora_b.shape[1], mod.lora_a.shape[0]
            base_norm = (
                base_norms[name]
                if base_norms and name in base_norms
                else _compute_base_weight_norm(mod.linear)
            )
            # 3D bilinear random sketch matrices (out_dim x 3) and (in_dim x 3)
            u_proj = _seeded_rademacher(f"{name}:U", (out_dim, 3))
            v_proj = _seeded_rademacher(f"{name}:V", (in_dim, 3))
            self.modules.append(
                {
                    "name": name,
                    "layer": layer_idx,
                    "proj": proj_name,
                    "group": group,
                    "mod": mod,
                    "base_norm": max(base_norm, 1e-8),
                    "u_proj": u_proj,
                    "v_proj": v_proj,
                }
            )

        mx.clear_cache()

        meta = {
            "run_id": self.run_dir.name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "model": config_dict.get("model"),
            "data": config_dict.get("data"),
            "adapter_path": config_dict.get("adapter_path"),
            "total_iters": self.total_iters,
            "num_lora_modules": len(self.modules),
            "num_layers": max((x["layer"] for x in self.modules), default=-1) + 1,
            "rank": config_dict.get("lora_parameters", {}).get("rank"),
            "scale": config_dict.get("lora_parameters", {}).get("scale"),
            "modules": [
                {
                    "name": x["name"],
                    "layer": x["layer"],
                    "proj": x["proj"],
                    "group": x["group"],
                    "base_norm": round(x["base_norm"], 4),
                }
                for x in self.modules
            ],
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _measure_parameter_space(self) -> dict[str, Any]:
        norm_sq_exprs = []
        proj_exprs = []
        for entry in self.modules:
            mod: LoRALinear = entry["mod"]
            a = mod.lora_a.astype(mx.float32)  # (in, r)
            b = mod.lora_b.astype(mx.float32)  # (r, out)
            scale = float(mod.scale)
            # Exact Frobenius norm^2 via trace identity: scale^2 * tr((B @ B.T) * (A.T @ A))
            bbt = b @ b.T
            ata = a.T @ a
            norm_sq = (scale * scale) * mx.sum(bbt * ata)
            # 3D bilinear sketch: sum_r ((V^T @ A) * (B @ U)^T) * scale -> (3,)
            bu = (b @ entry["u_proj"]).T  # (3, r)
            va = entry["v_proj"].T @ a  # (3, r)
            xyz = scale * mx.sum(bu * va, axis=1)
            norm_sq_exprs.append(norm_sq)
            proj_exprs.append(xyz)

        mx.eval(*(norm_sq_exprs + proj_exprs))

        total_norm_sq = 0.0
        attn_norm_sq = 0.0
        mlp_norm_sq = 0.0
        global_xyz = [0.0, 0.0, 0.0]
        attn_xyz = [0.0, 0.0, 0.0]
        mlp_xyz = [0.0, 0.0, 0.0]
        layer_xyz: dict[int, list[float]] = {}
        layer_norm_sq: dict[int, float] = {}
        module_metrics: list[dict[str, Any]] = []

        for entry, nsq_arr, xyz_arr in zip(self.modules, norm_sq_exprs, proj_exprs):
            nsq = max(float(nsq_arr.item()), 0.0)
            norm = math.sqrt(nsq)
            rel_pct = (norm / entry["base_norm"]) * 100.0
            xyz = [float(v) for v in xyz_arr.tolist()]

            total_norm_sq += nsq
            for k in range(3):
                global_xyz[k] += xyz[k]

            grp = entry["group"]
            if grp == "attn":
                attn_norm_sq += nsq
                for k in range(3):
                    attn_xyz[k] += xyz[k]
            elif grp == "mlp":
                mlp_norm_sq += nsq
                for k in range(3):
                    mlp_xyz[k] += xyz[k]

            lyr = entry["layer"]
            if lyr >= 0:
                if lyr not in layer_xyz:
                    layer_xyz[lyr] = [0.0, 0.0, 0.0]
                    layer_norm_sq[lyr] = 0.0
                layer_norm_sq[lyr] += nsq
                for k in range(3):
                    layer_xyz[lyr][k] += xyz[k]

            module_metrics.append(
                {
                    "layer": lyr,
                    "proj": entry["proj"],
                    "group": grp,
                    "delta_norm": round(norm, 6),
                    "rel_pct": round(rel_pct, 5),
                }
            )

        step_speed = 0.0
        if self.prev_xyz is not None:
            step_speed = math.sqrt(
                sum((global_xyz[k] - self.prev_xyz[k]) ** 2 for k in range(3))
            )
        self.prev_xyz = list(global_xyz)

        layers_summary = {
            str(lyr): {
                "xyz": [round(v, 5) for v in layer_xyz[lyr]],
                "delta_norm": round(math.sqrt(layer_norm_sq[lyr]), 5),
            }
            for lyr in sorted(layer_xyz)
        }

        return {
            "total_delta_norm": round(math.sqrt(total_norm_sq), 5),
            "attn_delta_norm": round(math.sqrt(attn_norm_sq), 5),
            "mlp_delta_norm": round(math.sqrt(mlp_norm_sq), 5),
            "xyz": [round(v, 5) for v in global_xyz],
            "attn_xyz": [round(v, 5) for v in attn_xyz],
            "mlp_xyz": [round(v, 5) for v in mlp_xyz],
            "step_displacement": round(step_speed, 5),
            "layers": layers_summary,
            "modules": module_metrics,
        }

    def _append_event(self, event: dict[str, Any]) -> None:
        with self.telemetry_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def on_train_loss_report(self, train_info: dict) -> None:
        if not self.recorded_step0 and self.modules:
            # Record step 0 origin (= frozen regular model before updates)
            snap0 = self._measure_parameter_space()
            self._append_event(
                {
                    "type": "param_snapshot",
                    "iteration": 0,
                    "param_space": snap0,
                }
            )
            self.recorded_step0 = True

        snap = self._measure_parameter_space() if self.modules else {}
        event = {
            "type": "train",
            "iteration": int(train_info.get("iteration", 0)),
            "train_loss": round(float(train_info.get("train_loss", 0.0)), 5),
            "learning_rate": float(train_info.get("learning_rate", 0.0)),
            "iterations_per_second": round(float(train_info.get("iterations_per_second", 0.0)), 3),
            "tokens_per_second": round(float(train_info.get("tokens_per_second", 0.0)), 2),
            "trained_tokens": int(train_info.get("trained_tokens", 0)),
            "peak_memory_gb": round(float(train_info.get("peak_memory", 0.0)), 2),
            "param_space": snap,
        }
        self._append_event(event)

    def on_val_loss_report(self, val_info: dict) -> None:
        event = {
            "type": "val",
            "iteration": int(val_info.get("iteration", 0)),
            "val_loss": round(float(val_info.get("val_loss", 0.0)), 5),
            "val_time": round(float(val_info.get("val_time", 0.0)), 2),
        }
        self._append_event(event)


def build_args(cli: argparse.Namespace) -> types.SimpleNamespace:
    cfg = dict(CONFIG_DEFAULTS)
    if cli.config and Path(cli.config).exists():
        loaded = yaml.safe_load(Path(cli.config).read_text(encoding="utf-8")) or {}
        cfg.update(loaded)

    if cli.smoke_test:
        cfg["iters"] = 30
        cfg["steps_per_report"] = 2
        cfg["steps_per_eval"] = 10
        cfg["val_batches"] = 5
        cfg["save_every"] = 15
        cfg["adapter_path"] = cli.adapter_path or "./adapters_smoke"
    else:
        if cli.iters is not None:
            cfg["iters"] = cli.iters
        if cli.adapter_path is not None:
            cfg["adapter_path"] = cli.adapter_path
        if cli.steps_per_report is not None:
            cfg["steps_per_report"] = cli.steps_per_report
        if cli.steps_per_eval is not None:
            cfg["steps_per_eval"] = cli.steps_per_eval

    if cli.batch_size is not None:
        cfg["batch_size"] = cli.batch_size
    if cli.grad_checkpoint:
        cfg["grad_checkpoint"] = True

    # Ensure cosine schedule length matches actual iters if overridden
    if isinstance(cfg.get("lr_schedule"), dict):
        sched = dict(cfg["lr_schedule"])
        args_list = list(sched.get("arguments", []))
        if len(args_list) >= 2:
            args_list[1] = cfg["iters"]
            sched["arguments"] = args_list
        if sched.get("warmup", 0) >= cfg["iters"]:
            sched["warmup"] = max(1, cfg["iters"] // 10)
        cfg["lr_schedule"] = sched

    cfg["train"] = True
    return types.SimpleNamespace(**cfg)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA training with live parameter-space visualizer telemetry.")
    parser.add_argument("-c", "--config", default="./lora_config.yaml")
    parser.add_argument("--run-name", default=None, help="Optional folder name under ./runs/")
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--steps-per-report", type=int, default=None)
    parser.add_argument("--steps-per-eval", type=int, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 30-step test (saves to ./adapters_smoke) to verify training & visualizer live.",
    )
    return parser.parse_args()


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    cli = parse_cli()
    args = build_args(cli)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cli.run_name or (f"smoke_{stamp}" if cli.smoke_test else f"run_{stamp}")
    runs_root = Path("./runs")
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (runs_root / "active_run.txt").write_text(run_name, encoding="utf-8")

    print(f"Loading model: {args.model}")
    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        trust_remote_code=args.trust_remote_code,
    )

    print(f"Loading dataset from: {args.data}")
    train_set, valid_set, _ = load_dataset(args, tokenizer)

    # Apply LoRA structure ahead of train_model? Wait: train_model applies linear_to_lora_layers,
    # so we replicate the freeze + linear_to_lora_layers prep? No, train_model itself calls
    # linear_to_lora_layers! Let's let a lightweight wrapper attach right after conversion,
    # or pre-measure base norms before train_model!
    # Wait: pre-converting here would double-wrap if train_model wraps again, so instead we
    # measure base norms on a shallow reference map? Even cleaner: compute base_norm map first,
    # then wrap? No: ParameterSpaceCallback needs references to the actual LoRALinear modules!
    # Let's check: in train_model, it calls linear_to_lora_layers, then calls train(..., training_callback).
    # We can subclass/hook OR simply convert here? No, train_model lines 247-252 ALWAYS call
    # linear_to_lora_layers unless fine_tune_type == 'full'.
    # So we attach on the first callback OR compute base norms cleanly by inspecting model before train_model
    # and matching module names on first on_val_loss_report / on_train_loss_report!
    base_norms: dict[str, float] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.QuantizedLinear)) and (
            "self_attn." in name or "mlp." in name
        ):
            base_norms[name] = _compute_base_weight_norm(mod)
    mx.clear_cache()

    cb = ParameterSpaceCallback(run_dir=run_dir, total_iters=args.iters)
    # Attach dynamically once train_model replaces Linear -> LoRALinear and starts callbacks
    orig_on_val = cb.on_val_loss_report
    orig_on_train = cb.on_train_loss_report

    def ensure_attached() -> None:
        if cb.modules:
            return
        cb.attach_model(model, vars(args), base_norms=base_norms)

    def wrapped_val(val_info: dict) -> None:
        ensure_attached()
        orig_on_val(val_info)

    def wrapped_train(train_info: dict) -> None:
        ensure_attached()
        orig_on_train(train_info)

    cb.on_val_loss_report = wrapped_val  # type: ignore[method-assign]
    cb.on_train_loss_report = wrapped_train  # type: ignore[method-assign]

    print(f"Telemetry streaming to: {cb.telemetry_path}")
    print(f"Adapters will save to:  {args.adapter_path}")
    train_model(args, model, train_set, valid_set, training_callback=cb)


if __name__ == "__main__":
    main()
