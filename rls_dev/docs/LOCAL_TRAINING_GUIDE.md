# Local Training Guide (M4 Mac)

Run real training locally on your M4 Mac to validate the RLS feature before cloud deployment.

## Quick Start

```bash
source .venv/bin/activate
./local_real_train.sh
```

This will:
1. Download 10 data shards (~1GB) if needed
2. Train baseline model (depth=8, 500 steps)
3. Train RLS model (same config)
4. Save logs for comparison

**Estimated time:** 1-2 hours total

## Configuration Options

Edit `local_real_train.sh` to adjust parameters:

### Fast Test (~20 minutes)
```bash
DEPTH=6
NUM_ITERATIONS=200
DEVICE_BATCH_SIZE=4
```

### Standard Test (~1-2 hours) - **Default**
```bash
DEPTH=8
NUM_ITERATIONS=500
DEVICE_BATCH_SIZE=2
```

### Thorough Test (~4-6 hours)
```bash
DEPTH=8
NUM_ITERATIONS=2000
DEVICE_BATCH_SIZE=2
```

### Larger Model (~8-12 hours)
```bash
DEPTH=10
NUM_ITERATIONS=1000
DEVICE_BATCH_SIZE=1  # May need to reduce if OOM
```

## What to Expect

### Performance
- **M4 Mac**: ~100-200ms per step for depth=8
- **Memory**: ~8-12GB for depth=8
- **RLS overhead**: ~2x slower (warmup passes)

### Results to Look For

1. **Training loss curve**: Should decrease over time
2. **Final loss**: Compare baseline vs RLS
3. **Gradient norms**: Check logs for learning signals
4. **No crashes**: Both models complete successfully

### Sample Output

```
Step 100/500 | train loss 3.2145 | val loss - | dt 150.23ms
Step 200/500 | train loss 2.8934 | val loss - | dt 148.91ms
...
Step 500/500 | train loss 2.1523 | val loss - | dt 147.02ms
```

## Understanding Results

### Baseline vs RLS Comparison

After both runs complete:

```bash
cd local_experiments

# Check final losses
tail -50 baseline_train.log | grep "train loss"
tail -50 rls_train.log | grep "train loss"
```

**What you're looking for:**
- Both models train without errors ✅
- Losses decrease over time ✅
- RLS final loss ≤ baseline (maybe, not guaranteed in short runs)

**Important:** Short local runs won't show definitive performance improvements. The goal is to verify:
1. The code works
2. No obvious regressions
3. Safe to test on expensive hardware

## Manual Single Run

To run just one model manually:

```bash
# Baseline
python -m scripts.base_train -- \
    --depth=8 \
    --device_batch_size=2 \
    --total_batch_size=2048 \
    --num_iterations=500 \
    --max_seq_len=512 \
    --device_type=mps \
    --recurrent_layer_state=False \
    --eval_every=-1 \
    --core_metric_every=-1

# With RLS
python -m scripts.base_train -- \
    --depth=8 \
    --device_batch_size=2 \
    --total_batch_size=2048 \
    --num_iterations=500 \
    --max_seq_len=512 \
    --device_type=mps \
    --recurrent_layer_state=True \
    --num_recurrence_warmup=1 \
    --eval_every=-1 \
    --core_metric_every=-1
```

## Monitoring Progress

### Watch logs in real-time

```bash
# Terminal 1: Run training
./local_real_train.sh

# Terminal 2: Monitor
tail -f local_experiments/baseline_train.log
# or
tail -f local_experiments/rls_train.log
```

### Check GPU/Memory usage

```bash
# Activity Monitor > GPU tab
# Should see Python process using GPU
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
DEVICE_BATCH_SIZE=1
```

Or reduce model size:
```bash
DEPTH=6
```

### Very Slow

MPS can be slower than CUDA. Expected speeds:
- depth=6: ~50-100ms/step
- depth=8: ~100-200ms/step
- depth=10: ~200-400ms/step

### "No module named X"

Make sure virtual environment is activated:
```bash
source .venv/bin/activate
```

### Training loss not decreasing

This can happen with very small runs. Try:
- More iterations (1000+)
- Slightly larger batch size
- Check that data downloaded correctly

## Data Management

### Download more data

```bash
# Download 20 shards (~2GB)
python -m nanochat.dataset -n 20

# Download 50 shards (~5GB)
python -m nanochat.dataset -n 50
```

More data = less overfitting, but 10 shards is enough for validation.

### Check downloaded data

```bash
ls -lh ~/.cache/nanochat/data
```

## Comparing Results

After both runs, create a simple comparison:

```bash
cd local_experiments

echo "=== BASELINE ==="
grep "Step.*train loss" baseline_train.log | tail -20

echo ""
echo "=== RLS ==="
grep "Step.*train loss" rls_train.log | tail -20
```

## Next Steps After Local Validation

### If local training succeeds:

**Option 1: Longer local run**
- Increase `NUM_ITERATIONS` to 2000-5000
- Add evaluation: `--eval_every=100`
- Run overnight

**Option 2: Single GPU cloud test**
- Rent single H100 (~$2-3/hour)
- Run depth=12-16 for 2000-5000 steps
- Should complete in 1-2 hours

**Option 3: Full speedrun**
- If confident, run full 8xH100 speedrun
- Edit `speedrun.sh` to add `--recurrent_layer_state=True`
- Cost: ~$100, Time: ~4 hours

## Cost Comparison

| Setup | Hardware | Time | Cost | Quality |
|-------|----------|------|------|---------|
| Local fast | M4 Mac | 20min | Free | Sanity check |
| Local standard | M4 Mac | 1-2hr | Free | Basic validation |
| Local thorough | M4 Mac | 4-6hr | Free | Good signal |
| Single GPU | H100 | 1-2hr | $2-6 | Strong signal |
| Full speedrun | 8×H100 | 4hr | $100 | Production quality |

## Recommended Path

1. **Start:** `./local_real_train.sh` (default config, ~1-2 hours)
2. **If successful:** Run overnight with 2000 iterations
3. **If still promising:** Single GPU cloud test
4. **If validated:** Full speedrun

This minimizes risk while providing validation at each step.
