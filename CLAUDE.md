# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

nanochat is a full-stack implementation of a ChatGPT-like LLM in a minimal, hackable codebase. It trains end-to-end on a single 8XH100 node, covering tokenization, pretraining, finetuning, evaluation, inference, and web serving. This is designed to be the capstone project for LLM101n course.

## Common Commands

### Environment Setup
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv sync --extra gpu  # For GPU systems
uv sync --extra cpu  # For CPU/MPS systems
source .venv/bin/activate

# Build the Rust BPE tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Running Training Scripts

**Full speedrun (~$100, 4 hours on 8XH100):**
```bash
bash speedrun.sh
# Or in a screen session with logging:
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

**Larger model (~$1000, 41.6 hours):**
```bash
bash run1000.sh
```

**Individual training stages (distributed):**
```bash
# Pretraining (depth=20 is default for speedrun)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train

# Supervised finetuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Reinforcement learning (optional)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

**Single GPU (omit torchrun, 8x slower):**
```bash
python -m scripts.base_train -- --depth=20
```

### Evaluation and Inference

```bash
# Evaluate base model on CORE tasks
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Evaluate chat model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Chat via CLI (leave out -p for interactive mode)
python -m scripts.chat_cli -p "Why is the sky blue?"

# Serve web UI (ChatGPT-like interface)
python -m scripts.chat_web
```

### Tokenizer

```bash
# Train tokenizer
python -m scripts.tok_train --max_chars=2000000000

# Evaluate tokenizer compression
python -m scripts.tok_eval
```

### Data Management

```bash
# Download pretraining data shards (each ~250M chars, ~100MB compressed)
python -m nanochat.dataset -n 240  # Download 240 shards
```

### Testing

```bash
# Run tokenizer tests
python -m pytest tests/test_rustbpe.py -v -s
```

### Report Generation

```bash
# Reset and initialize report
python -m nanochat.report reset

# Generate full report (creates report.md)
python -m nanochat.report generate
```

## Architecture Overview

### Training Pipeline Stages

1. **Tokenizer Training**: Custom Rust BPE tokenizer (`rustbpe/`) trained on pretraining data
2. **Base Model Pretraining**: GPT model trained on raw text from FineWeb dataset
3. **Midtraining**: Teaches conversation special tokens, tool use, and multiple choice formatting
4. **Supervised Finetuning (SFT)**: Domain adaptation on conversation datasets
5. **Reinforcement Learning (RL)**: Optional stage, currently only for GSM8K math problems

### Core Components

**Model Architecture (`nanochat/gpt.py`):**
- Transformer with rotary embeddings (RoPE) instead of positional embeddings
- QK normalization in attention layers
- Untied embedding weights (separate token embedding and lm_head)
- ReLU² activation in MLP
- RMSNorm without learnable parameters
- No bias in linear layers
- Multi-Query Attention (MQA) support for efficient inference

**Data Loading (`nanochat/dataloader.py`):**
- Streaming data loader from Parquet files
- Distributed: each rank processes different shards
- Tokenizes on-the-fly with multithreading
- Yields batches of (inputs, targets) with shape (B, T)

**Optimizers:**
- `nanochat/muon.py`: Distributed Muon optimizer for matrix parameters (attention/MLP weights)
- `nanochat/adamw.py`: Distributed AdamW for embedding/unembedding parameters
- Default split: Muon for most params, AdamW for embeddings with different LRs

**Inference Engine (`nanochat/engine.py`):**
- Efficient inference with KV cache
- Supports sampling strategies (temperature, top-k, top-p)
- Tool execution capability (Python code execution)

**Configuration System (`nanochat/configurator.py`):**
- Alternative to argparse that allows config file + CLI overrides
- Usage: `python script.py config/file.py --param=value`
- Overrides global variables directly (intentionally minimal and hackable)

### Evaluation Tasks (`tasks/`)

All tasks inherit from base classes in `tasks/common.py`:
- `arc.py`: ARC-Challenge and ARC-Easy science questions
- `gsm8k.py`: Grade school math problems
- `humaneval.py`: Python coding challenges
- `mmlu.py`: Multiple choice questions across broad topics
- `smoltalk.py`: Conversational dataset from HuggingFace
- `spellingbee.py`: Letter counting/spelling tasks
- `customjson.py`: Create custom tasks from JSONL conversation files

### Directory Structure

- `nanochat/`: Core library modules (model, optimizers, data loading, tokenizer interface)
- `scripts/`: Training and inference entry points
- `rustbpe/`: Rust-based BPE tokenizer for fast parallel training/encoding
- `tasks/`: Evaluation task definitions
- `tests/`: Test suite (primarily tokenizer tests)
- `dev/`: Development utilities and example scripts

### Key Hyperparameters

**Model sizing (based on depth parameter):**
- `num_layers = depth`
- `model_dim = depth * 64` (aspect ratio of 64)
- `num_heads = max(1, (model_dim + 127) // 128)` (head dimension of 128)

**Scaling laws:**
- Follows Chinchilla: tokens = 20 × parameters
- Data requirement: `params * 20 * 4.8 / 250e6` shards (assuming 4.8 chars/token compression)

**Memory management:**
- Adjust `--device_batch_size` to avoid OOM (32→16→8→4→2→1)
- Lower batch sizes trigger automatic gradient accumulation
- For larger models, use `--depth=26` or `--depth=32` with reduced batch size

## Key Implementation Details

**Distributed Training:**
- Uses PyTorch DDP via `torchrun`
- Each rank processes different data shards in lockstep
- Gradient accumulation compensates for smaller per-device batches

**Wandb Integration:**
- Set `WANDB_RUN=name` environment variable to enable logging
- Special value "dummy" disables wandb (default)
- Must run `wandb login` first

**Device Support:**
- Primary: CUDA (H100, A100)
- CPU and MPS (Apple Silicon) supported via `device_type` parameter
- Auto-detection: CUDA > MPS > CPU

**Data Storage:**
- Default artifacts directory: `~/.cache/nanochat` (set via `NANOCHAT_BASE_DIR`)
- Checkpoints saved as `base_d{depth}.pt`, `mid.pt`, `sft.pt`, `rl.pt`

## Customization

**Personality/Identity:**
- Modify synthetic data generation (see `dev/gen_synthetic_data.py`)
- Mix custom data into midtraining via `identity_conversations.jsonl`
- Reference: [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139)

**Adding Capabilities:**
- Create custom tasks in `tasks/` using base classes
- Add to training mixture via `TaskMixture` or `TaskSequence`
- Reference: [Guide: counting r in strawberry](https://github.com/karpathy/nanochat/discussions/164)

**Adjusting Model Size:**
- Change `--depth` parameter (e.g., d20→d26→d32)
- Reduce `--device_batch_size` if OOM
- Download more data shards proportionally

## Philosophy

nanochat prioritizes:
- **Minimalism**: Single cohesive codebase, not a framework with heavy configuration
- **Readability**: No if-then-else monsters or giant config objects
- **Hackability**: Direct global variable manipulation, exec-based configuration
- **End-to-end**: Complete pipeline from tokenization to web serving
- **Accessibility**: Works on budgets < $1000 with clear cost/performance tradeoffs
