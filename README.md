# Apple Silicon Local LLM Comparison & LoRA Observatory

Zero-framework-bloat local tools built with pure Python + Apple **MLX** (`mlx-lm`) for comparing local models side-by-side and watching LoRA fine-tuning happen live in parameter space and loss curves.

> No React/Node build step required — each UI is a self-contained local HTTP server with an animated dark aurora interface and built-in onboarding guide.

---

## Tools Included

| File | Purpose |
| :--- | :--- |
| [`compare_ui.py`](compare_ui.py) | Local side-by-side comparison web app with SSE live streaming, **Keep both in RAM**, **Stream both live**, **Copy**, **Undergrad Mode**, and markdown export (**Save individual** / **Save to log**). |
| [`train_visual.py`](train_visual.py) | Wraps `mlx_lm.lora` training and streams step-by-step per-layer/per-module LoRA weight displacement (`Delta W = B @ A`) and train/val loss to `./runs/<run_id>/`. |
| [`visualizer.py`](visualizer.py) | Live **LoRA Parameter Space & Loss Observatory**: 3D PCA trajectory vs. the frozen Regular Model `(0, 0, 0)`, 36×7 Layer×Module heatmap, train/val loss curve (EMA + log scale), top-moving matrices, and time-scrubber replay. |
| [`lora_config.yaml`](lora_config.yaml) | Clean default LoRA training config tuned for unified-memory Macs. |
| [`benchmark.py`](benchmark.py) | CLI head-to-head throughput & response benchmark between two local MLX models. |
| [`examples/sample_valid.jsonl`](examples/sample_valid.jsonl) | 3 sample chat-format prompts so `compare_ui.py` works out of the box even before you add your own dataset. |

---

## Setup (Apple Silicon Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download or point to any two MLX models on disk or Hugging Face repo IDs (for example under `./models/`), and place your chat JSONL dataset at `./data/train.jsonl` and `./data/valid.jsonl`:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## 1. Side-by-Side Comparison UI

```bash
.venv/bin/python compare_ui.py \
  --model-a mlx-community/Qwen3-8B-8bit \
  --model-b mlx-community/Qwen2.5-14B-Instruct-4bit \
  --prompts ./examples/sample_valid.jsonl
```

If you already store local weights in `./models/Qwen3-8B-MLX-8bit` and `./models/Qwen3.8-27B-4bit`, running `.venv/bin/python compare_ui.py` picks them up automatically (and falls back to `./data/valid.jsonl` or `./examples/sample_valid.jsonl`).

Then open `http://127.0.0.1:8765/`.

### Features
- **Sequential vs Keep Both in RAM vs Stream Both Live**
- **Undergrad Mode** to test your own answer against the models and dataset reference answer
- **Save individual** (`./saved_comparisons/comparison_*.md`) or **Save to log** (`./saved_comparisons/comparison_log.md`)

---

## 2. Live LoRA Training & Parameter-Space Visualizer

Open two terminals:

**Terminal 1 — Start the Observatory UI**
```bash
.venv/bin/python visualizer.py
# Open http://127.0.0.1:8780/
```

**Terminal 2 — Run a quick smoke test (no full dataset required if pointed to `./examples`) or full fine-tune**
```bash
# Fast 6-step smoke test to watch weights move immediately:
.venv/bin/python train_visual.py --smoke-test

# Full training run using lora_config.yaml and ./data/{train,valid}.jsonl:
.venv/bin/python train_visual.py
```

---

## 3. CLI Benchmark

```bash
.venv/bin/python benchmark.py --limit 5
```

---

## License

MIT — fork, modify, and use freely.
