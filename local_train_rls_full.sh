#!/bin/bash

# Full epoch RLS validation run
# This script trains to completion using all available data
#
# Weights & Biases integration:
#   - Automatically logs both runs to W&B (if wandb is configured)
#   - Run names: rls-side-tokens-{timestamp} and baseline-{timestamp}
#   - Set WANDB_DISABLE=1 to disable wandb logging

set -e          # Exit on error
set -o pipefail # Catch errors in pipes
set -u          # Error on undefined variables

echo "=========================================="
echo "Full Epoch RLS Validation"
echo "=========================================="
echo ""

# Generate unique identifier for this experiment run
RUN_ID=$(date +%Y%m%d-%H%M%S)

# Configuration options
# Set NUM_SHARDS to control training length:
#   4 shards   = 208M tokens = 407k steps = full epoch (no data repetition)
#   71 shards  = 3.7B tokens = 7.25M steps = Chinchilla optimal
NUM_SHARDS=4    # Change to 71 for Chinchilla optimal

# Model configuration
DEPTH=12          # ~186M params (12 layers, 768 dim)
MAX_SEQ_LEN=512

# H100-optimized batch sizes
# Use same batch size for both runs for fair comparison
# Batch size affects optimization dynamics, so we need apples-to-apples
# RLS layer-0 uses 2x memory (T×2T attention), so we reduce from 256→128
DEVICE_BATCH=128   # Fits both baseline and RLS comfortably
TOTAL_BATCH=65536   # 128 * 512 (single gradient accumulation step)

# Calculate number of iterations for full epoch
# Each shard has ~250M chars, compression ~4.8 chars/token
TOKENS_PER_SHARD=$((250000000 / 5))  # ~50M tokens per shard (conservative estimate)
TOTAL_TOKENS=$((NUM_SHARDS * TOKENS_PER_SHARD))
NUM_ITERS=$((TOTAL_TOKENS / TOTAL_BATCH))

echo "Configuration:"
echo "  Data shards: $NUM_SHARDS"
echo "  Total tokens: ~$((TOTAL_TOKENS / 1000000))M"
echo "  Training iterations: $NUM_ITERS"
echo "  Depth: $DEPTH (~186M parameters, 12 layers)"
echo "  Sequence length: $MAX_SEQ_LEN"
echo "  Device batch size: $DEVICE_BATCH"
echo "  Total batch size: $TOTAL_BATCH tokens"
if [ "$NUM_SHARDS" -eq 4 ]; then
    echo "  Training regime: Full epoch (1 pass through data)"
    echo "  Estimated time: ~25 min on H100 (batch_size=128)"
else
    echo "  Training regime: Chinchilla optimal"
    echo "  Estimated time: ~7 hours on H100 (batch_size=128)"
fi
echo ""

# Setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
source .venv/bin/activate

echo "Step 1: Download training data ($NUM_SHARDS shards)"
echo "--------------------------------------------------"
# Check if data already exists
EXISTING_SHARDS=$(ls -1 $NANOCHAT_BASE_DIR/data/*.parquet 2>/dev/null | wc -l || echo 0)
EXISTING_SHARDS=$(echo $EXISTING_SHARDS | tr -d ' ')

if [ "$EXISTING_SHARDS" -ge "$NUM_SHARDS" ]; then
    echo "✓ Found $EXISTING_SHARDS shards (need $NUM_SHARDS)"
else
    echo "Downloading $NUM_SHARDS shards..."
    echo "This may take a while (~$((NUM_SHARDS * 100))MB total)"
    python -m nanochat.dataset -n $NUM_SHARDS
fi
echo ""

echo "Step 2: Train tokenizer (if needed)"
echo "------------------------------------"
if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "✓ Tokenizer already trained"
else
    echo "Training tokenizer on 1B characters (~10-15 minutes)..."
    python -m scripts.tok_train --max_chars=1000000000
fi
echo ""

# Create output directory
mkdir -p local_rls_experiments_full

echo "=========================================="
echo "RUN 1: With Recurrent Layer State"
echo "=========================================="
date
echo ""

# Set wandb run name for RLS training
export WANDB_RUN="rls-side-tokens-d${DEPTH}-${RUN_ID}"
echo "W&B Run: $WANDB_RUN"
echo ""

if ! python -u -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH \
    --total_batch_size=$TOTAL_BATCH \
    --num_iterations=$NUM_ITERS \
    --recurrent_layer_state=True \
    --num_recurrence_warmup=1 \
    --tokenizer_threads=1 \
    --eval_every=-1 \
    --core_metric_every=-1 \
    --sample_every=10000 \
    --log_every=100 \
    2>&1 | tee local_rls_experiments_full/rls.log; then
    echo ""
    echo "❌ ERROR: RLS training failed!"
    echo "Check local_rls_experiments_full/rls.log for details"
    exit 1
fi

echo ""
echo "✓ RLS complete!"
echo ""

echo "=========================================="
echo "RUN 2: Baseline (no RLS)"
echo "=========================================="
date
echo ""

# Set wandb run name for baseline training
export WANDB_RUN="baseline-d${DEPTH}-${RUN_ID}"
echo "W&B Run: $WANDB_RUN"
echo ""

if ! python -u -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH \
    --total_batch_size=$TOTAL_BATCH \
    --num_iterations=$NUM_ITERS \
    --recurrent_layer_state=False \
    --tokenizer_threads=1 \
    --eval_every=-1 \
    --core_metric_every=-1 \
    --sample_every=10000 \
    --log_every=100 \
    2>&1 | tee local_rls_experiments_full/baseline.log; then
    echo ""
    echo "❌ ERROR: Baseline training failed!"
    echo "Check local_rls_experiments_full/baseline.log for details"
    exit 1
fi

echo ""
echo "✓ Baseline complete!"
echo ""

echo "=========================================="
echo "RESULTS"
echo "=========================================="
date
echo ""

echo "Extracting final losses..."
echo ""

echo "--- BASELINE ---"
grep "step.*train loss" local_rls_experiments_full/baseline.log | tail -10
echo ""

echo "--- RLS ---"
grep "step.*train loss" local_rls_experiments_full/rls.log | tail -10
echo ""

# Extract final loss values
BASELINE_LOSS=$(grep "step" local_rls_experiments_full/baseline.log | grep "loss:" | tail -1 | grep -oE "loss: [0-9]+\.[0-9]+" | awk '{print $2}')
RLS_LOSS=$(grep "step" local_rls_experiments_full/rls.log | grep "loss:" | tail -1 | grep -oE "loss: [0-9]+\.[0-9]+" | awk '{print $2}')

BASELINE_VAL=$(grep "Validation bpb" local_rls_experiments_full/baseline.log | tail -1 | grep -oE "[0-9]+\.[0-9]+")
RLS_VAL=$(grep "Validation bpb" local_rls_experiments_full/rls.log | tail -1 | grep -oE "[0-9]+\.[0-9]+")

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Training iterations: $NUM_ITERS"
echo "Total tokens: ~$((TOTAL_TOKENS / 1000000))M"
echo ""
echo "Final Training Loss:"
echo "  Baseline: $BASELINE_LOSS"
echo "  RLS:      $RLS_LOSS"
echo ""
echo "Final Validation BPB:"
echo "  Baseline: $BASELINE_VAL"
echo "  RLS:      $RLS_VAL"
echo ""
echo "Logs saved in local_rls_experiments_full/"
echo "  - baseline.log"
echo "  - rls.log"
echo ""

if [ "$NUM_SHARDS" -eq 4 ]; then
    echo "Completed: Full epoch through available data"
else
    echo "Completed: Chinchilla optimal training"
fi
echo ""
