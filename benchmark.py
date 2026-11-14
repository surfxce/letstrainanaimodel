#!/usr/bin/env python3
"""Sequential side-by-side benchmark for MLX models (e.g., 8B vs 27B).

Loads one model at a time, runs the same prompt set, records throughput and
peak unified memory, frees memory cleanly, then runs the second model and
writes both a Markdown comparison report and raw JSON results.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


DEFAULT_PROMPTS: list[dict[str, str]] = [
    {
        "prompt": (
            "Distinguish acute resolving neuroinflammation from chronic "
            "neuroinflammation, and explain the dual physiological versus "
            "pathological roles of microglia in the CNS."
        ),
        "reference": "",
    },
    {
        "prompt": (
            "Walk through the initiation and enzymatic convertase sequence of "
            "the Classical Complement Pathway."
        ),
        "reference": "",
    },
    {
        "prompt": (
            "Explain the mechanism of the Alternative Complement Pathway, "
            "including 'tickover' activation, convertase formation, and why it "
            "acts as the primary amplification loop for the entire complement system."
        ),
        "reference": "",
    },
]


@dataclass
class PromptResult:
    index: int
    prompt: str
    reference: str
    response: str
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    wall_time_s: float
    finish_reason: str | None


def load_prompts(path: str | None, limit: int | None) -> list[dict[str, str]]:
    if path is None:
        items = list(DEFAULT_PROMPTS)
    else:
        p = Path(path)
        items = []
        if p.suffix == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "messages" in obj:
                        sys_msgs = [
                            m["content"]
                            for m in obj["messages"]
                            if m.get("role") == "system"
                        ]
                        user_msgs = [
                            m["content"]
                            for m in obj["messages"]
                            if m.get("role") == "user"
                        ]
                        asst_msgs = [
                            m["content"]
                            for m in obj["messages"]
                            if m.get("role") == "assistant"
                        ]
                        if not user_msgs:
                            continue
                        items.append(
                            {
                                "system": sys_msgs[-1] if sys_msgs else "",
                                "prompt": user_msgs[-1],
                                "reference": asst_msgs[-1] if asst_msgs else "",
                            }
                        )
                    elif "prompt" in obj:
                        items.append(
                            {
                                "system": obj.get("system", ""),
                                "prompt": obj["prompt"],
                                "reference": obj.get("reference", obj.get("completion", "")),
                            }
                        )
        else:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append({"prompt": line, "reference": ""})
    if limit is not None:
        items = items[:limit]
    return items


def format_chat_prompt(
    tokenizer: Any,
    user_prompt: str,
    system_prompt: str | None,
    enable_thinking: bool | None,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if not hasattr(tokenizer, "apply_chat_template"):
        return user_prompt

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def run_model_benchmark(
    label: str,
    model_path: str,
    adapter_path: str | None,
    prompts: list[dict[str, str]],
    system_prompt: str | None,
    max_tokens: int,
    temp: float,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    print(f"\n=== Loading [{label}]: {model_path} ===")
    if adapter_path:
        print(f"    Adapter: {adapter_path}")

    mx.reset_peak_memory()
    t_load_start = time.perf_counter()
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    load_time_s = time.perf_counter() - t_load_start
    print(f"Loaded in {load_time_s:.2f}s")

    sampler = make_sampler(temp=temp)
    results: list[PromptResult] = []

    for idx, item in enumerate(prompts, start=1):
        effective_system = system_prompt if system_prompt is not None else (item.get("system") or None)
        formatted = format_chat_prompt(
            tokenizer, item["prompt"], effective_system, enable_thinking
        )
        t0 = time.perf_counter()
        chunks: list[str] = []
        last_resp = None
        for resp in stream_generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            chunks.append(resp.text)
            last_resp = resp
        wall_s = time.perf_counter() - t0

        text = "".join(chunks).strip()
        pr = PromptResult(
            index=idx,
            prompt=item["prompt"],
            reference=item["reference"],
            response=text,
            prompt_tokens=last_resp.prompt_tokens if last_resp else 0,
            generation_tokens=last_resp.generation_tokens if last_resp else 0,
            prompt_tps=round(last_resp.prompt_tps, 2) if last_resp else 0.0,
            generation_tps=round(last_resp.generation_tps, 2) if last_resp else 0.0,
            peak_memory_gb=round(last_resp.peak_memory, 2) if last_resp else 0.0,
            wall_time_s=round(wall_s, 2),
            finish_reason=last_resp.finish_reason if last_resp else None,
        )
        results.append(pr)
        print(
            f"  [{idx}/{len(prompts)}] "
            f"{pr.generation_tokens} tok | "
            f"prefill {pr.prompt_tps:.1f} t/s | "
            f"gen {pr.generation_tps:.1f} t/s | "
            f"peak {pr.peak_memory_gb:.2f} GB | "
            f"{pr.wall_time_s:.2f}s"
        )

    overall_peak_gb = round(mx.get_peak_memory() / (1024**3), 2)

    del model, tokenizer
    gc.collect()
    mx.clear_cache()

    avg_prompt_tps = round(
        sum(r.prompt_tps for r in results) / max(len(results), 1), 2
    )
    avg_gen_tps = round(
        sum(r.generation_tps for r in results) / max(len(results), 1), 2
    )

    return {
        "label": label,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "load_time_s": round(load_time_s, 2),
        "overall_peak_memory_gb": overall_peak_gb,
        "avg_prompt_tps": avg_prompt_tps,
        "avg_generation_tps": avg_gen_tps,
        "results": [asdict(r) for r in results],
    }


def write_markdown_report(
    out_path: Path, small_run: dict[str, Any], large_run: dict[str, Any]
) -> None:
    lines: list[str] = [
        "# Side-by-Side Model Benchmark",
        "",
        f"- **Model A:** `{small_run['label']}` (`{small_run['model_path']}`)"
        + (f" + adapter `{small_run['adapter_path']}`" if small_run["adapter_path"] else ""),
        f"- **Model B:** `{large_run['label']}` (`{large_run['model_path']}`)",
        "",
        "## Summary Metrics",
        "",
        "| Metric | " + small_run["label"] + " | " + large_run["label"] + " |",
        "| :--- | ---: | ---: |",
        f"| Load time (s) | {small_run['load_time_s']} | {large_run['load_time_s']} |",
        f"| Peak memory (GB) | {small_run['overall_peak_memory_gb']} | {large_run['overall_peak_memory_gb']} |",
        f"| Avg prefill (tok/s) | {small_run['avg_prompt_tps']} | {large_run['avg_prompt_tps']} |",
        f"| Avg decode (tok/s) | {small_run['avg_generation_tps']} | {large_run['avg_generation_tps']} |",
        "",
        "## Responses",
        "",
    ]

    for a, b in zip(small_run["results"], large_run["results"]):
        lines.extend(
            [
                f"### Prompt {a['index']}",
                "",
                f"> {a['prompt']}",
                "",
            ]
        )
        if a["reference"]:
            lines.extend(
                [
                    "**Reference Answer:**",
                    "",
                    a["reference"],
                    "",
                ]
            )
        lines.extend(
            [
                f"**{small_run['label']}** "
                f"({a['generation_tokens']} tok, {a['generation_tps']} tok/s, {a['wall_time_s']}s):",
                "",
                a["response"],
                "",
                f"**{large_run['label']}** "
                f"({b['generation_tokens']} tok, {b['generation_tps']} tok/s, {b['wall_time_s']}s):",
                "",
                b["response"],
                "",
                "---",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential side-by-side benchmark for 8B vs 27B MLX models."
    )
    parser.add_argument(
        "--small-model",
        default="./models/Qwen3-8B-MLX-8bit",
        help="Path or HF repo for the smaller model.",
    )
    parser.add_argument(
        "--small-label",
        default="Qwen3-8B-8bit",
        help="Display label for the smaller model.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter directory for --small-model.",
    )
    parser.add_argument(
        "--large-model",
        default="./models/Qwen3.8-27B-4bit",
        help="Path or HF repo for the larger control model.",
    )
    parser.add_argument(
        "--large-label",
        default="Qwen3.8-27B-4bit",
        help="Display label for the larger control model.",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Optional .jsonl (chat messages or prompt) or .txt file of prompts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of prompts to run.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt prepended to each chat request.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens per prompt.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic greedy decoding).",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass enable_thinking to chat template when supported (default: False).",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results",
        help="Directory for comparison Markdown and JSON outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts, args.limit)
    if not prompts:
        raise SystemExit("No prompts loaded.")

    small_run = run_model_benchmark(
        label=args.small_label,
        model_path=args.small_model,
        adapter_path=args.adapter_path,
        prompts=prompts,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        enable_thinking=args.thinking,
    )
    large_run = run_model_benchmark(
        label=args.large_label,
        model_path=args.large_model,
        adapter_path=None,
        prompts=prompts,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
        enable_thinking=args.thinking,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_{stamp}.json"
    md_path = out_dir / f"benchmark_{stamp}.md"

    payload = {
        "timestamp": stamp,
        "small": small_run,
        "large": large_run,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(md_path, small_run, large_run)

    print(f"\nSaved JSON report:     {json_path}")
    print(f"Saved Markdown report: {md_path}")


if __name__ == "__main__":
    main()
