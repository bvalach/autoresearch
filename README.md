# autoresearch-mlx-lora

> Fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) adapted for **Apple Silicon** using **MLX LoRA fine-tuning** instead of PyTorch/CUDA from-scratch training.

## What's different

| | Original (Karpathy) | This fork |
|---|---|---|
| **Hardware** | Single NVIDIA GPU (H100) | Apple Silicon (M-series) |
| **Framework** | PyTorch + CUDA + Flash Attention 3 | MLX (Apple's ML framework) |
| **Approach** | Train a GPT from scratch | LoRA fine-tune an existing model |
| **Base model** | N/A (trains from random init) | Qwen3.5-0.8B |
| **Agent** | Claude / Codex (cloud API) | Qwen3.5:9b via Ollama (fully local) |
| **Dataset** | climbmix-400b | TinyStories |
| **Metric** | val_bpb | val_bpb (same) |
| **Time budget** | 5 minutes per experiment | 5 minutes per experiment (same) |

The core idea is identical: give an AI agent a training setup and let it experiment autonomously overnight. It modifies the configuration, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats.

The key difference is that everything runs **locally on a MacBook** — no cloud APIs, no NVIDIA GPU, no API costs. The researcher agent (Qwen3.5:9b) runs via Ollama, and the training target (Qwen3.5-0.8B) is fine-tuned with LoRA adapters using MLX.

## How it works

```
orchestrate.py (the loop)
    │
    ├─→ Ollama API (local) → Qwen3.5:9b
    │     reads: program.md + train.py + results.tsv
    │     proposes: changes to train.py
    │
    ├─→ Applies changes to train.py
    ├─→ git commit
    ├─→ Runs LoRA training via mlx-lm (5 min budget)
    ├─→ Evaluates BPB on validation set
    ├─→ keep (new best) / discard (revert)
    └─→ LOOP forever
```

Four files that matter:

| File | Role | Who edits it |
|---|---|---|
| `prepare.py` | Downloads model + dataset, evaluation metric | Nobody (fixed) |
| `train.py` | LoRA config + training hyperparameters | The agent (9B) |
| `program.md` | Instructions and context for the agent | You (the human) |
| `orchestrate.py` | Autonomous experiment loop | Nobody (fixed) |

## Quick start

**Requirements:** Apple Silicon Mac (M1/M2/M3/M4/M5), 16+ GB unified memory, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.ai) with Qwen3.5:9b.

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Ollama and pull the agent model (if needed)
# https://ollama.ai → download and install
ollama pull qwen3.5:9b

# 3. Install dependencies (creates .venv automatically)
uv sync

# 4. Download base model and prepare dataset (~2 min)
uv run prepare.py

# 5. Run a single baseline training to verify setup (~5 min)
uv run train.py

# 6. Launch autonomous research (runs forever until you stop it)
uv run orchestrate.py
```

## What the agent can tune

The agent modifies `train.py` each iteration. The search space includes:

- **LoRA architecture:** rank (4-64), scale (10-40), dropout, number of layers
- **Optimization:** learning rate, optimizer (Adam/AdamW/Muon), LR schedule
- **Data:** batch size, gradient accumulation, max sequence length
- **Random seed**

## Design choices

- **LoRA, not full training.** On Apple Silicon with 24 GB unified memory, LoRA fine-tuning of a 0.8B model is fast (~5 min/iteration) and fits comfortably. Full pre-training of even a small GPT would be too slow to iterate meaningfully.
- **Fully local.** No API calls, no cloud dependency. The 9B agent model runs in Ollama alongside the 0.8B training target. Both share the M-series unified memory.
- **Same loop logic as Karpathy.** 5-minute time budget, BPB metric, git-based keep/discard. The `results.tsv` format is compatible.
- **TinyStories dataset.** Following Karpathy's recommendation for small models — lower entropy data yields meaningful results with fewer parameters.

## Hardware notes

Tested on Apple M5 (24 GB). Memory usage during training:

- Qwen3.5:9b in Ollama: ~6.6 GB
- Qwen3.5-0.8B LoRA training: ~2-4 GB
- Total: ~10 GB, leaving headroom for macOS

On 16 GB Macs: consider using a smaller agent (Qwen3:8b or Qwen3-0.6B) or quantizing the agent model.

## Future directions

- **Pi as scheduler:** Offload the orchestration loop to a Raspberry Pi, keeping the Mac's full memory for training.
- **Multi-model tournaments:** Run different base models (Qwen3-0.6B vs Qwen3.5-0.8B vs Qwen3-1.7B) and let them compete.
- **Dataset exploration:** Let the agent choose from multiple datasets, not just TinyStories.
- **Adapter stacking:** Accumulate LoRA adapters across experiments instead of training from scratch each time.

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for the original autoresearch concept
- [Apple MLX team](https://github.com/ml-explore) for MLX and mlx-lm
- [Qwen team](https://huggingface.co/Qwen) for the Qwen3.5 model family
- [Ollama](https://ollama.ai) for making local LLM inference simple

## License

MIT
