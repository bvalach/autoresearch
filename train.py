"""
Time-budgeted MLX LoRA training for autoresearch-mlx-lora.

This script is designed for Apple Silicon. It fine-tunes a local MLX-converted
Qwen3.5-2B-Base model with LoRA adapters on TinyStories for a fixed wall-clock
budget, then evaluates the resulting adapter with validation BPB.

Usage:
    uv run train.py
"""

from __future__ import annotations

import os
import shutil
import time
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map

from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.tuner.trainer import default_loss, evaluate, grad_checkpoint, iterate_batches
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import load, save_config

from prepare import EVAL_SAMPLES, TIME_BUDGET, evaluate_bpb, get_data_dir, get_model_path

# ---------------------------------------------------------------------------
# LoRA Configuration
# ---------------------------------------------------------------------------

LORA_CONFIG = {
    "rank": 16,
    "dropout": 0.0,
    "scale": 20.0,
}
LORA_NUM_LAYERS = 16

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-5
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 512
OPTIMIZER = "adamw"
LR_SCHEDULE = "cosine"
MAX_ITERS = 600
VAL_BATCHES = 12
STEPS_PER_EVAL = 25
STEPS_PER_REPORT = 10
SAVE_EVERY = 100
SEED = 42

# Fixed safety settings for 2B on unified memory.
GRAD_CHECKPOINT = True
FINE_TUNE_TYPE = "lora"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(get_model_path())
DATA_DIR = Path(get_data_dir())
ADAPTER_DIR = ROOT_DIR / "adapters"
ADAPTER_FILE = ADAPTER_DIR / "adapters.safetensors"


# ---------------------------------------------------------------------------
# Runtime Helpers
# ---------------------------------------------------------------------------

def ensure_artifacts_exist():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MLX model at {MODEL_PATH}. Run `uv run prepare.py` first."
        )

    required = [DATA_DIR / "train.jsonl", DATA_DIR / "valid.jsonl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared dataset files:\n"
            + "\n".join(missing)
            + "\nRun `uv run prepare.py --data-only` first."
        )


def reset_adapter_dir():
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)


def build_learning_rate():
    if LR_SCHEDULE is None:
        return LEARNING_RATE
    if LR_SCHEDULE == "cosine":
        # Keep a small LR floor so the last updates still move the adapters.
        return optim.schedulers.cosine_decay(
            LEARNING_RATE,
            MAX_ITERS,
            end=LEARNING_RATE * 0.1,
        )
    raise ValueError(f"Unsupported LR_SCHEDULE: {LR_SCHEDULE!r}")


def build_optimizer():
    learning_rate = build_learning_rate()
    name = OPTIMIZER.lower()
    if name == "adam":
        return optim.Adam(learning_rate=learning_rate)
    if name == "adamw":
        return optim.AdamW(learning_rate=learning_rate)
    if name == "muon":
        return optim.Muon(learning_rate=learning_rate)
    if name == "sgd":
        return optim.SGD(learning_rate=learning_rate)
    if name == "adafactor":
        return optim.Adafactor(learning_rate=learning_rate)
    raise ValueError(f"Unsupported OPTIMIZER: {OPTIMIZER!r}")


def current_learning_rate(optimizer):
    value = optimizer.learning_rate
    return value.item() if hasattr(value, "item") else float(value)


def save_adapter_weights(model, checkpoint_name: str | None = None):
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(ADAPTER_FILE), adapter_weights)
    if checkpoint_name is not None:
        mx.save_safetensors(str(ADAPTER_DIR / checkpoint_name), adapter_weights)


def build_run_config():
    return {
        "model": str(MODEL_PATH),
        "data": str(DATA_DIR),
        "fine_tune_type": FINE_TUNE_TYPE,
        "num_layers": LORA_NUM_LAYERS,
        "batch_size": BATCH_SIZE,
        "iters": MAX_ITERS,
        "val_batches": VAL_BATCHES,
        "learning_rate": LEARNING_RATE,
        "steps_per_report": STEPS_PER_REPORT,
        "steps_per_eval": STEPS_PER_EVAL,
        "adapter_path": str(ADAPTER_DIR),
        "save_every": SAVE_EVERY,
        "max_seq_length": MAX_SEQ_LENGTH,
        "grad_checkpoint": GRAD_CHECKPOINT,
        "grad_accumulation_steps": GRAD_ACCUM_STEPS,
        "lr_schedule": LR_SCHEDULE,
        "optimizer": OPTIMIZER,
        "seed": SEED,
        "lora_parameters": dict(LORA_CONFIG),
    }


def load_training_components():
    tokenizer_config = {"trust_remote_code": True}
    model, tokenizer = load(str(MODEL_PATH), tokenizer_config=tokenizer_config)

    dataset_config = SimpleNamespace(mask_prompt=False, text_feature="text")
    train_set, valid_set, _ = load_local_dataset(DATA_DIR, tokenizer, dataset_config)

    if len(train_set) == 0:
        raise ValueError("Prepared train split is empty.")
    if len(valid_set) == 0:
        raise ValueError("Prepared validation split is empty.")

    if LORA_NUM_LAYERS > len(model.layers):
        raise ValueError(
            f"LORA_NUM_LAYERS={LORA_NUM_LAYERS} exceeds model depth {len(model.layers)}."
        )

    model.freeze()
    linear_to_lora_layers(model, LORA_NUM_LAYERS, dict(LORA_CONFIG))
    print_trainable_parameters(model)
    return model, tokenizer, CacheDataset(train_set), CacheDataset(valid_set)


def train_with_time_budget(model, optimizer, train_dataset, valid_dataset):
    if mx.metal.is_available():
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])

    world = mx.distributed.init()
    rank = world.rank()

    if GRAD_CHECKPOINT:
        grad_checkpoint(model.layers[0])

    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch, prev_grad, do_update):
        (loss_value, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            grad = tree_map(lambda x, y: x + y, grad, prev_grad)

        if do_update:
            grad = average_gradients(grad)
            if GRAD_ACCUM_STEPS > 1:
                grad = tree_map(lambda x: x / GRAD_ACCUM_STEPS, grad)
            optimizer.update(model, grad)
            grad = None

        return loss_value, toks, grad

    model.train()
    start_time = time.perf_counter()
    last_val_loss = float("inf")
    peak_mem_gb = 0.0
    grad_accum = None
    losses = 0
    n_tokens = 0
    report_steps = 0
    trained_tokens = 0
    report_train_time = 0.0
    completed_iters = 0

    batch_iter = iterate_batches(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        loop=True,
        comm_group=world,
    )

    print(f"Starting training with wall-clock budget {TIME_BUDGET}s")
    for it, batch in zip(range(1, MAX_ITERS + 1), batch_iter):
        if valid_dataset and (it == 1 or it % STEPS_PER_EVAL == 0):
            val_t0 = time.perf_counter()
            last_val_loss = evaluate(
                model=model,
                dataset=valid_dataset,
                batch_size=BATCH_SIZE,
                num_batches=VAL_BATCHES,
                max_seq_length=MAX_SEQ_LENGTH,
                loss=default_loss,
                iterate_batches=iterate_batches,
            )
            model.train()
            val_dt = time.perf_counter() - val_t0
            if rank == 0:
                print(
                    f"Iter {it}: Val loss {last_val_loss:.3f}, "
                    f"Val took {val_dt:.3f}s",
                    flush=True,
                )

        train_t0 = time.perf_counter()
        loss_value, toks, grad_accum = step(
            batch,
            grad_accum,
            it % GRAD_ACCUM_STEPS == 0,
        )

        losses += loss_value
        n_tokens += toks
        report_steps += 1
        mx.eval(state, losses, n_tokens, grad_accum)
        report_train_time += time.perf_counter() - train_t0
        peak_mem_gb = max(peak_mem_gb, mx.get_peak_memory() / 1e9)
        completed_iters = it

        if it % STEPS_PER_REPORT == 0 or it == MAX_ITERS:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= max(report_steps, 1) * world.size()
            token_count = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            lr = current_learning_rate(optimizer)
            it_per_sec = report_steps / report_train_time if report_train_time else 0.0
            tok_per_sec = float(token_count) / report_train_time if report_train_time else 0.0
            trained_tokens += token_count
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {lr:.3e}, "
                    f"It/sec {it_per_sec:.3f}, "
                    f"Tokens/sec {tok_per_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem_gb:.3f} GB",
                    flush=True,
                )
            losses = 0
            n_tokens = 0
            report_steps = 0
            report_train_time = 0.0

        if it % SAVE_EVERY == 0 and rank == 0:
            save_adapter_weights(model, checkpoint_name=f"{it:07d}_adapters.safetensors")
            print(f"Iter {it}: Saved adapter weights to {ADAPTER_FILE}.", flush=True)

        elapsed = time.perf_counter() - start_time
        if it % GRAD_ACCUM_STEPS == 0 and elapsed >= TIME_BUDGET:
            break

    if rank == 0:
        save_adapter_weights(model)
        print(f"Saved final weights to {ADAPTER_FILE}.", flush=True)

    training_seconds = time.perf_counter() - start_time
    return last_val_loss, peak_mem_gb, completed_iters, training_seconds


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ensure_artifacts_exist()
    reset_adapter_dir()

    np.random.seed(SEED)
    mx.random.seed(SEED)

    run_config = build_run_config()
    save_config(dict(run_config), ADAPTER_DIR / "adapter_config.json")

    wall_t0 = time.perf_counter()
    model, tokenizer, train_dataset, valid_dataset = load_training_components()
    optimizer = build_optimizer()

    val_loss, peak_mem_gb, num_steps, training_seconds = train_with_time_budget(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )

    model.eval()
    val_bpb = evaluate_bpb(
        model,
        tokenizer,
        data_path=DATA_DIR / "valid.jsonl",
        max_samples=EVAL_SAMPLES,
    )

    total_seconds = time.perf_counter() - wall_t0
    peak_mem_mb = peak_mem_gb * 1024

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_mem_mb:      {peak_mem_mb:.1f}")
    print(f"peak_mem_gb:      {peak_mem_gb:.3f}")
    print(f"num_steps:        {num_steps}")
    print(f"eval_samples:     {EVAL_SAMPLES}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
