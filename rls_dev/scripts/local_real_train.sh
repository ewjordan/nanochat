#!/bin/bash
# Real training run on M4 Mac with MPS
# Compares baseline vs RLS on a small but realistic setup

set -e

echo "=========================================="
echo "Local Real Training Run"
echo "=========================================="
echo ""

# Check for data
DATA_DIR=~/.cache/nanochat/data
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "No training data found. Downloading 10 shards (~1GB)..."
    echo "This will take a few minutes..."
    python -m nanochat.dataset -n 10
    echo ""
fi

# Check for tokenizer
TOKENIZER_DIR=~/.cache/nanochat/tokenizer
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ]; then
    echo "No tokenizer found. Training tokenizer on limited data..."
    echo "This will take 10-20 minutes..."
    echo ""
    # Use subset of data for faster tokenizer training
    python -m scripts.tok_train --max_chars=500000000
    echo ""
    echo "Tokenizer training complete!"
    echo ""
fi

# Configuration for M4 Mac
# depth=6: ~23M params (manageable on Mac)
# device_batch_size=4: Small enough to fit in unified memory
# num_iterations=1000: Quick but meaningful
# max_seq_len=512: Shorter sequences to save memory

DEPTH=8
DEVICE_BATCH_SIZE=2
NUM_ITERATIONS=500
MAX_SEQ_LEN=512
TOTAL_BATCH_SIZE=2048  # Small for Mac

echo "Configuration:"
echo "  Model depth: $DEPTH (~13M parameters)"
echo "  Device batch size: $DEVICE_BATCH_SIZE"
echo "  Total batch size: $TOTAL_BATCH_SIZE (tokens)"
echo "  Training iterations: $NUM_ITERATIONS"
echo "  Sequence length: $MAX_SEQ_LEN"
echo "  Device: MPS (M4 Mac)"
echo "  Eval: disabled for speed"
echo ""
echo "This will take approximately 1-2 hours total."
echo ""

# Create output directory
mkdir -p local_experiments
cd local_experiments

echo "=========================================="
echo "RUN 1: Baseline (no RLS)"
echo "=========================================="
echo ""

python -m scripts.base_train -- \
    --depth=$DEPTH \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=$NUM_ITERATIONS \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_type=mps \
    --recurrent_layer_state=False \
    --eval_every=-1 \
    --core_metric_every=-1 \
    2>&1 | tee baseline_train.log

echo ""
echo "Baseline complete!"
echo ""
echo "=========================================="
echo "RUN 2: With Recurrent Layer State"
echo "=========================================="
echo ""

python -m scripts.base_train -- \
    --depth=$DEPTH \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=$NUM_ITERATIONS \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_type=mps \
    --recurrent_layer_state=True \
    --num_recurrence_warmup=1 \
    --eval_every=-1 \
    --core_metric_every=-1 \
    2>&1 | tee rls_train.log

echo ""
echo "=========================================="
echo "DONE!"
echo "=========================================="
echo ""
echo "Check the logs:"
echo "  baseline_train.log - Baseline run"
echo "  rls_train.log - RLS run"
echo ""
echo "Look for final loss values and compare."
