"""
Autoresearch LoRA fine-tuning script. Apple Silicon, single-file.
This is the file the agent modifies. Everything below is fair game.

Usage: uv run train.py
"""

import os
import time
import json
import types

from prepare import TIME_BUDGET, get_model_path, get_data_dir, evaluate_bpb

# ---------------------------------------------------------------------------
# LoRA Configuration (agent: modify these freely)
# ---------------------------------------------------------------------------

LORA_CONFIG = {
    "rank": 8,              # low-rank dimension (try 4, 8, 16, 32, 64)
    "dropout": 0.0,         # dropout in LoRA layers (try 0.0, 0.05, 0.1)
    "scale": 20.0,          # scaling factor (alpha/rank, try 10-40)
}

LORA_NUM_LAYERS = -1        # how many layers to apply LoRA to (-1 = all)

# ---------------------------------------------------------------------------
# Training Hyperparameters (agent: modify these freely)
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-4        # try 5e-5, 1e-4, 2e-4, 5e-4
BATCH_SIZE = 1              # per-step batch size (try 1, 2, 4)
GRAD_ACCUM_STEPS = 8        # gradient accumulation (effective batch = BATCH_SIZE * this)
MAX_SEQ_LENGTH = 256        # max tokens per sample (try 256, 512, 1024)
OPTIMIZER = "adamw"         # optimizer: "adam" or "adamw"
LR_SCHEDULE = "cosine"      # schedule: "cosine" or None
STEPS_PER_EVAL = 50         # evaluate every N steps
STEPS_PER_REPORT = 10       # log every N steps
SEED = 42                   # random seed

# Max iterations — upper bound, the trainer runs for exactly this many steps.
# Estimate: ~0.5-2s per step on M5 with batch_size=1, so 150-600 steps in 5 min.
MAX_ITERS = 250

# ---------------------------------------------------------------------------
# Setup: load model and tokenizer
# ---------------------------------------------------------------------------

t_start = time.time()

model_path = get_model_path()
data_dir = get_data_dir()

print(f"Model: {model_path}")
print(f"Data: {data_dir}")
print(f"Time budget: {TIME_BUDGET}s")
print(f"LoRA config: {LORA_CONFIG}")
print(f"LR: {LEARNING_RATE}, batch: {BATCH_SIZE}, grad_accum: {GRAD_ACCUM_STEPS}")
print(f"Max seq length: {MAX_SEQ_LENGTH}, optimizer: {OPTIMIZER}")
print()

# ---------------------------------------------------------------------------
# Training via mlx-lm LoRA
# ---------------------------------------------------------------------------

from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train as lora_train
from mlx_lm.tuner.datasets import load_dataset as load_lora_dataset, CacheDataset
from mlx_lm.tuner.utils import linear_to_lora_layers
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

mx.random.seed(SEED)

# Load base model and tokenizer
model, tokenizer = load(model_path)

# Freeze base model, then apply LoRA (which auto-unfreezes LoRA params)
model.freeze()
linear_to_lora_layers(model, num_layers=LORA_NUM_LAYERS, config=LORA_CONFIG)

# Count parameters
def count_params(params):
    total = 0
    if isinstance(params, dict):
        for v in params.values():
            total += count_params(v)
    elif isinstance(params, list):
        for v in params:
            total += count_params(v)
    elif hasattr(params, 'size'):
        total += params.size
    return total

total_p = count_params(model.parameters())
trainable_p = count_params(model.trainable_parameters())
print(f"Total parameters: {total_p:,}")
print(f"Trainable parameters (LoRA): {trainable_p:,} ({100*trainable_p/total_p:.2f}%)")
print()

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# mlx-lm's load_dataset expects an args-like object with specific attributes
dataset_args = types.SimpleNamespace(
    data=data_dir,
    train=True,
    test=False,
    hf_dataset=False,
)
train_set, val_set, _ = load_lora_dataset(dataset_args, tokenizer)
print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

# ---------------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------------

if LR_SCHEDULE == "cosine":
    lr = optim.schedulers.cosine_decay(init=LEARNING_RATE, decay_steps=MAX_ITERS)
else:
    lr = LEARNING_RATE

if OPTIMIZER == "adam":
    opt = optim.Adam(learning_rate=lr)
elif OPTIMIZER == "adamw":
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)
else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

adapter_dir = os.path.join(os.path.dirname(__file__) or ".", "adapters")
os.makedirs(adapter_dir, exist_ok=True)
adapter_file = os.path.join(adapter_dir, "adapters.safetensors")

training_args = TrainingArgs(
    batch_size=BATCH_SIZE,
    iters=MAX_ITERS,
    val_batches=25,
    steps_per_report=STEPS_PER_REPORT,
    steps_per_eval=STEPS_PER_EVAL,
    max_seq_length=MAX_SEQ_LENGTH,
    adapter_file=adapter_file,
    grad_checkpoint=True,
    grad_accumulation_steps=GRAD_ACCUM_STEPS,
)

print(f"Starting LoRA training ({MAX_ITERS} iters, budget: {TIME_BUDGET}s) ...")
print("---")

t_start_training = time.time()

model.train()
lora_train(
    model=model,
    optimizer=opt,
    train_dataset=CacheDataset(train_set),
    val_dataset=CacheDataset(val_set),
    args=training_args,
)

t_end_training = time.time()
training_seconds = t_end_training - t_start_training

print()
print(f"Training completed in {training_seconds:.1f}s")

# ---------------------------------------------------------------------------
# Final evaluation: BPB (the metric that matters)
# ---------------------------------------------------------------------------

print("Evaluating BPB on validation set ...")
model.eval()
val_bpb = evaluate_bpb(model, tokenizer)

# Memory estimate (Apple Silicon unified memory)
# On macOS, ru_maxrss is in bytes
import resource
try:
    peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
except Exception:
    peak_mem_mb = 0.0

# ---------------------------------------------------------------------------
# Final summary (same format as Karpathy's original for compatibility)
# ---------------------------------------------------------------------------

t_end = time.time()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_mem_mb:      {peak_mem_mb:.1f}")
print(f"total_params_M:   {total_p / 1e6:.1f}")
print(f"trainable_params: {trainable_p:,}")
print(f"lora_rank:        {LORA_CONFIG['rank']}")
print(f"learning_rate:    {LEARNING_RATE}")
print(f"optimizer:        {OPTIMIZER}")
