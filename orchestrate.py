"""
Autonomous research orchestrator for autoresearch-mlx-lora.

Runs an infinite experiment loop where a local LLM (Qwen3.5:9b via Ollama)
acts as the researcher: it reads the current state, proposes changes to
train.py, and decides whether to keep or discard based on val_bpb.

Usage:
    uv run orchestrate.py                    # default: Qwen3.5:9b
    uv run orchestrate.py --agent qwen3:8b   # use a different Ollama model

The orchestrator:
  1. Reads program.md, train.py, and results.tsv for context
  2. Asks the agent LLM what to try next
  3. Applies the proposed changes to train.py
  4. Commits, trains (5 min), evaluates BPB
  5. Keeps or discards based on improvement
  6. Loops forever until manually stopped

Requires: Ollama running locally with the agent model pulled.
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
import tempfile

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
DEFAULT_AGENT = "qwen3.5:9b"
TRAIN_TIMEOUT = 600  # 10 min hard timeout (budget is 5 min + startup)
RESULTS_FILE = "results.tsv"
TRAIN_FILE = "train.py"
PROGRAM_FILE = "program.md"

# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

def ollama_chat(model, messages, temperature=0.7):
    """Send a chat request to Ollama and return the response text."""
    import requests

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 512,
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def check_ollama(model):
    """Verify Ollama is running and the model is available."""
    import requests
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        # Check for exact match or prefix match (e.g. "qwen3.5:9b" in "qwen3.5:9b-...")
        found = any(model in m or m.startswith(model) for m in models)
        if not found:
            print(f"Warning: model '{model}' not found in Ollama. Available: {models}")
            print(f"Pull it with: ollama pull {model}")
            return False
        return True
    except requests.ConnectionError:
        print("Error: Ollama is not running. Start it with: ollama serve")
        return False


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args):
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
    )
    return result.stdout.strip(), result.returncode


def git_commit_short():
    """Return current short commit hash."""
    out, _ = git("rev-parse", "--short", "HEAD")
    return out


def git_commit_and_get_hash(message):
    """Stage train.py, commit, return short hash. Returns (hash, committed)."""
    git("add", TRAIN_FILE)
    _, rc = git("commit", "-m", message)
    committed = rc == 0
    return git_commit_short(), committed


def git_revert_last():
    """Revert the last commit (keep changes staged)."""
    git("reset", "--hard", "HEAD~1")


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------

def init_results():
    """Create results.tsv with header if it doesn't exist."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def read_results():
    """Read results.tsv and return as string."""
    if not os.path.exists(RESULTS_FILE):
        return "(no results yet)"
    with open(RESULTS_FILE) as f:
        return f.read()


def append_result(commit, val_bpb, memory_gb, status, description):
    """Append a row to results.tsv."""
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


def get_best_bpb():
    """Return the best (lowest) val_bpb from results, or None."""
    if not os.path.exists(RESULTS_FILE):
        return None
    best = None
    with open(RESULTS_FILE) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3] != "crash":
                bpb = float(parts[1])
                if bpb > 0 and (best is None or bpb < best):
                    best = bpb
    return best


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training():
    """
    Run train.py, capture output to run.log.
    Returns (val_bpb, peak_mem_mb, success).
    """
    log_path = "run.log"

    try:
        result = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True, text=True,
            timeout=TRAIN_TIMEOUT,
            cwd=os.path.dirname(__file__) or ".",
        )

        # Write combined output to log
        with open(log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- stderr ---\n")
                f.write(result.stderr)

        if result.returncode != 0:
            return 0.0, 0.0, False

        # Parse results from output
        val_bpb = 0.0
        peak_mem = 0.0
        for line in result.stdout.split("\n"):
            if line.startswith("val_bpb:"):
                val_bpb = float(line.split(":")[1].strip())
            elif line.startswith("peak_mem_mb:"):
                peak_mem = float(line.split(":")[1].strip())

        return val_bpb, peak_mem, val_bpb > 0

    except subprocess.TimeoutExpired:
        with open(log_path, "w") as f:
            f.write("TIMEOUT: training exceeded 10 minute hard limit\n")
        return 0.0, 0.0, False


# ---------------------------------------------------------------------------
# Agent interaction
# ---------------------------------------------------------------------------

def build_agent_prompt(train_py_content, results_text, experiment_num):
    """Build the prompt for the agent LLM.

    Asks for ONLY the config values (not the full file) to minimize
    output tokens and keep response times fast with a local 9B model.
    """

    # Read program.md for context
    with open(PROGRAM_FILE) as f:
        program = f.read()

    best_bpb = get_best_bpb()
    best_str = f"{best_bpb:.6f}" if best_bpb else "(no baseline yet)"

    # Extract just the config section from train.py for context
    config_lines = []
    in_config = False
    for line in train_py_content.split("\n"):
        if "LoRA Configuration" in line or "Training Hyperparameters" in line:
            in_config = True
        elif in_config and line.startswith("# ------"):
            if config_lines:  # end of second config section
                break
            continue
        if in_config:
            config_lines.append(line)
    config_section = "\n".join(config_lines)

    return f"""You are an autonomous ML researcher minimizing val_bpb via LoRA fine-tuning.

Current config:
{config_section}

Results so far:
{results_text}
Best val_bpb: {best_str} | Experiment #{experiment_num}

Knobs: rank (4-64), dropout (0-0.2), scale (10-40), lora_num_layers (-1=all),
learning_rate (1e-5 to 1e-3), batch_size (1-4), grad_accum_steps (1-16),
max_seq_length (128-1024), optimizer (adam/adamw), lr_schedule (cosine/None),
max_iters (100-500).

Propose ONE change. Respond in EXACTLY this format (no extra text):
DESCRIPTION: <one line>
CHANGES:
VARIABLE_NAME = new_value
VARIABLE_NAME = new_value

Example:
DESCRIPTION: increase LoRA rank from 8 to 16
CHANGES:
LORA_CONFIG["rank"] = 16

/no_think"""


# Config variables that the agent is allowed to modify
ALLOWED_CONFIGS = {
    'LORA_CONFIG["rank"]', 'LORA_CONFIG["dropout"]', 'LORA_CONFIG["scale"]',
    'LORA_NUM_LAYERS', 'LEARNING_RATE', 'BATCH_SIZE', 'GRAD_ACCUM_STEPS',
    'MAX_SEQ_LENGTH', 'OPTIMIZER', 'LR_SCHEDULE', 'STEPS_PER_EVAL',
    'STEPS_PER_REPORT', 'SEED', 'MAX_ITERS',
}


def parse_agent_response(response):
    """Extract description and variable changes from agent response."""
    description = "unknown change"
    changes = {}

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("DESCRIPTION:"):
            description = line[len("DESCRIPTION:"):].strip()
        elif "=" in line and not line.startswith("#"):
            # Parse VARIABLE = value lines
            parts = line.split("=", 1)
            if len(parts) == 2:
                var = parts[0].strip()
                val = parts[1].strip()
                if var in ALLOWED_CONFIGS:
                    changes[var] = val

    return description, changes


def apply_changes(train_py_path, changes):
    """Apply variable changes to train.py by text replacement.
    Returns True if changes were applied successfully."""
    with open(train_py_path) as f:
        content = f.read()

    original = content
    for var, new_val in changes.items():
        if var.startswith('LORA_CONFIG["'):
            # Handle dict entries: LORA_CONFIG["rank"] = 8
            key = var.split('"')[1]
            # Match the key in the dict literal
            pattern = rf'("{key}":\s*)([^,\n}}]+)'
            replacement = rf'\g<1>{new_val}'
            content = re.sub(pattern, replacement, content)
        else:
            # Handle top-level variables: LEARNING_RATE = 1e-4
            pattern = rf'^({re.escape(var)}\s*=\s*)(.+)$'
            replacement = rf'\g<1>{new_val}'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    if content == original:
        return False

    with open(train_py_path, "w") as f:
        f.write(content)
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous LoRA research orchestrator")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="Ollama model for the agent")
    parser.add_argument("--branch", default=None, help="Git branch name (auto-generated if not set)")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline and stop")
    args = parser.parse_args()

    print("=" * 60)
    print("autoresearch-mlx-lora orchestrator")
    print(f"Agent: {args.agent} (via Ollama)")
    print("=" * 60)
    print()

    # Check Ollama
    if not check_ollama(args.agent):
        sys.exit(1)

    # Create branch
    if args.branch:
        tag = args.branch
    else:
        tag = time.strftime("%b%d").lower()
    branch_name = f"autoresearch/{tag}"

    current_branch, _ = git("branch", "--show-current")
    if current_branch != branch_name:
        git("checkout", "-b", branch_name)
        print(f"Created branch: {branch_name}")
    else:
        print(f"Already on branch: {branch_name}")

    # Initialize results
    init_results()

    # ---------------------------------------------------------------------------
    # Experiment loop
    # ---------------------------------------------------------------------------
    experiment_num = 0

    while True:
        experiment_num += 1
        print()
        print(f"{'=' * 60}")
        print(f"Experiment #{experiment_num}")
        print(f"{'=' * 60}")

        if experiment_num == 1:
            # First run: baseline (no changes)
            description = "baseline"
            print("Running baseline (no changes to train.py) ...")
        else:
            # Ask the agent what to try
            print(f"Asking {args.agent} for next experiment ...")
            with open(TRAIN_FILE) as f:
                train_content = f.read()

            prompt = build_agent_prompt(train_content, read_results(), experiment_num)
            messages = [{"role": "user", "content": prompt}]

            try:
                response = ollama_chat(args.agent, messages)
            except Exception as e:
                print(f"Agent error: {e}")
                print("Retrying in 10s ...")
                time.sleep(10)
                continue

            description, changes = parse_agent_response(response)

            if not changes:
                print(f"Could not parse agent response. Skipping.")
                print(f"Raw response:\n{response[:500]}...")
                continue

            # Apply changes
            print(f"Applying: {description}")
            for var, val in changes.items():
                print(f"  {var} = {val}")
            if not apply_changes(TRAIN_FILE, changes):
                print("Warning: no changes were applied (regex didn't match)")
                continue

        # Commit
        commit_hash, committed = git_commit_and_get_hash(f"exp #{experiment_num}: {description}")
        if committed:
            print(f"Committed: {commit_hash}")
        else:
            print(f"No changes to commit (using current HEAD: {commit_hash})")

        # Train
        print(f"Training (budget: 5 min) ...")
        t0 = time.time()
        val_bpb, peak_mem_mb, success = run_training()
        dt = time.time() - t0
        print(f"Finished in {dt:.0f}s")

        if not success:
            print(f"CRASH — logging and reverting")
            append_result(commit_hash, 0.0, 0.0, "crash", description)
            if committed:
                git_revert_last()
            continue

        memory_gb = peak_mem_mb / 1024
        print(f"val_bpb: {val_bpb:.6f} | memory: {memory_gb:.1f} GB")

        # Decide keep or discard
        best = get_best_bpb()
        if best is None or val_bpb < best:
            status = "keep"
            print(f"KEEP — new best! (previous: {best})")
        else:
            status = "discard"
            print(f"DISCARD — no improvement (best: {best:.6f})")

        append_result(commit_hash, val_bpb, memory_gb, status, description)

        if status == "discard" and committed:
            git_revert_last()

        if args.baseline_only and experiment_num == 1:
            print("\nBaseline complete. Stopping (--baseline-only).")
            break

        print(f"\nResults so far:")
        print(read_results())


if __name__ == "__main__":
    main()
