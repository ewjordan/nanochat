#!/bin/bash

# Local RLS validation run on M4 Mac
# Based on dev/runcpu.sh but compares baseline vs RLS

set -e

echo "=========================================="
echo "Local RLS Validation Run (M4 Mac)"
echo "=========================================="
echo ""

# Setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
source .venv/bin/activate

echo "Step 1: Download training data (4 shards, ~400MB)"
echo "--------------------------------------------------"
# Check if data already exists
if [ -d "$NANOCHAT_BASE_DIR/data" ] && [ "$(ls -A $NANOCHAT_BASE_DIR/data 2>/dev/null | wc -l)" -ge 4 ]; then
    echo "✓ Data already downloaded"
else
    python -m nanochat.dataset -n 4
fi
echo ""

echo "Step 2: Train tokenizer (if needed)"
echo "------------------------------------"
# Check if tokenizer exists
if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "✓ Tokenizer already trained"
else
    echo "Training tokenizer on 1B characters (~10-15 minutes)..."
    python -m scripts.tok_train --max_chars=1000000000
fi
echo ""

# Configuration - tuned for M4 Mac
DEPTH=6           # ~7M params
MAX_SEQ_LEN=512   # Shorter for memory
DEVICE_BATCH=1    # Single sequence at a time
TOTAL_BATCH=512   # Small total batch
NUM_ITERS=200     # Quick but meaningful

echo "Configuration:"
echo "  Depth: $DEPTH (~7M parameters)"
echo "  Sequence length: $MAX_SEQ_LEN"
echo "  Device batch size: $DEVICE_BATCH"
echo "  Total batch size: $TOTAL_BATCH tokens"
echo "  Training iterations: $NUM_ITERS"
echo "  Estimated time: 30-60 minutes total"
echo ""

# Create output directory
mkdir -p local_rls_experiments
cd local_rls_experiments

echo "=========================================="
echo "RUN 1: Baseline (no RLS)"
echo "=========================================="
date
echo ""

python -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH \
    --total_batch_size=$TOTAL_BATCH \
    --num_iterations=$NUM_ITERS \
    --recurrent_layer_state=False \
    --eval_every=-1 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    2>&1 | tee baseline.log

echo ""
echo "✓ Baseline complete!"
echo ""

echo "=========================================="
echo "RUN 2: With Recurrent Layer State"
echo "=========================================="
date
echo ""

python -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH \
    --total_batch_size=$TOTAL_BATCH \
    --num_iterations=$NUM_ITERS \
    --recurrent_layer_state=True \
    --num_recurrence_warmup=1 \
    --eval_every=-1 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    2>&1 | tee rls.log

echo ""
echo "✓ RLS complete!"
echo ""

echo "=========================================="
echo "RESULTS"
echo "=========================================="
date
echo ""

echo "Extracting final losses..."
echo ""

echo "--- BASELINE ---"
grep "step.*train loss" baseline.log | tail -10
echo ""

echo "--- RLS ---"
grep "step.*train loss" rls.log | tail -10
echo ""

# Extract final loss values
BASELINE_LOSS=$(grep "step.*train loss" baseline.log | tail -1 | grep -oE "train loss [0-9]+\.[0-9]+" | awk '{print $3}')
RLS_LOSS=$(grep "step.*train loss" rls.log | tail -1 | grep -oE "train loss [0-9]+\.[0-9]+" | awk '{print $3}')

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Baseline final loss: $BASELINE_LOSS"
echo "RLS final loss:      $RLS_LOSS"
echo ""
echo "Logs saved in local_rls_experiments/"
echo "  - baseline.log"
echo "  - rls.log"
echo ""
echo "Next steps:"
echo "  - If both trained successfully: Feature works! ✅"
echo "  - For better signal: Increase NUM_ITERS to 1000-2000"
echo "  - For cloud testing: Use depth=12-20 on GPU"
echo ""
