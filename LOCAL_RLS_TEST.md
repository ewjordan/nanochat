# Local RLS Testing Guide

Quick guide to validate the Recurrent Layer State feature locally on your M4 Mac before cloud deployment.

## What This Does

Runs a complete training comparison:
1. **Baseline model**: Standard nanochat (depth=6, ~7M params)
2. **RLS model**: Same model with recurrent layer state enabled

Both use real FineWeb training data and the same hyperparameters for fair comparison.

## Quick Start

```bash
source .venv/bin/activate
./local_train_rls_test.sh
```

**Time:** 30-60 minutes total
**Output:** Logs in `local_rls_experiments/` directory

## First Run Setup

The script will automatically:
1. Download 4 data shards (~400MB) if needed
2. Train tokenizer (~10-15 minutes) if needed
3. Run baseline training (200 iterations)
4. Run RLS training (200 iterations)
5. Compare results

## What to Expect

### During Training

You'll see output like:
```
step    10/200 | train loss 8.1234 | val loss - | dt 89.23ms | ...
step    20/200 | train loss 7.8523 | val loss - | dt 87.91ms | ...
...
```

- **Loss should decrease** over time
- **dt (time per step)**: 50-150ms for baseline, 100-300ms for RLS
- **RLS is ~2x slower**: Expected due to warmup passes

### Results

At the end you'll see:
```
SUMMARY
Baseline final loss: 5.2134
RLS final loss:      5.1987
```

**What this tells you:**
- ✅ Both models trained successfully → Feature works!
- Loss comparison → Preliminary signal (not conclusive with 200 steps)

## Adjusting Parameters

Edit the script to change:

### Faster Test (~15 minutes)
```bash
NUM_ITERS=100
```

### More Thorough (~2 hours)
```bash
NUM_ITERS=1000
DEPTH=8  # ~13M params
```

### Overnight Run (~6-8 hours)
```bash
NUM_ITERS=2000
DEPTH=8
```

## Monitoring Progress

### Watch in real-time
```bash
# Terminal 1: Run script
./local_train_rls_test.sh

# Terminal 2: Monitor
tail -f local_rls_experiments/baseline.log
# or
tail -f local_rls_experiments/rls.log
```

### Check memory usage
Open Activity Monitor → GPU tab → Look for Python process

## Understanding Results

### Success Criteria
1. **No crashes**: Both models complete all iterations
2. **Loss decreases**: Training loss goes down over time
3. **Reasonable final loss**: Should be < 6.0 for 200 iterations

### Performance Comparison

**Short runs (200 steps):** Not enough to show meaningful differences
- Focus on: Does it work without errors?

**Longer runs (1000+ steps):** Better signal
- Compare final losses
- Check loss curves (plot from logs)

## Troubleshooting

### Out of Memory
```bash
# Reduce depth
DEPTH=4

# Or reduce batch size (already at 1)
# May need to adjust total_batch_size too
TOTAL_BATCH=256
```

### Tokenizer training takes forever
- Normal on M4 Mac: 10-20 minutes
- Only happens once (cached afterward)

### "No module named X"
```bash
source .venv/bin/activate
```

### Very slow training
- Expected on MPS vs CUDA
- M4 Pro/Max faster than M4 base
- Can run overnight for better results

## Interpreting Results

### Both models train successfully
✅ **Code works!** Safe to test on cloud

### RLS loss slightly better
✅ **Promising sign** but not conclusive
→ Run longer test or cloud validation

### RLS loss slightly worse
⚠️ **Not necessarily bad** - warmup overhead matters more in short runs
→ Need longer training to see real impact

### About the same
✓ **Normal for short runs**
→ Extend to 1000+ iterations for better signal

## Next Steps

### After successful local run:

**1. Longer local validation** (recommended)
```bash
# Edit script:
NUM_ITERS=1000

./local_train_rls_test.sh
# Run overnight
```

**2. Single GPU cloud test**
```bash
# On H100/A100 instance
python -m scripts.base_train \
    --depth=12 \
    --num_iterations=2000 \
    --recurrent_layer_state=True
```

**3. Full speedrun** (when confident)
```bash
# 8xH100, ~$100, 4 hours
# Add --recurrent_layer_state=True to speedrun.sh
WANDB_RUN=rls_test bash speedrun.sh
```

## Files Created

```
local_rls_experiments/
├── baseline.log          # Full training log for baseline
├── rls.log              # Full training log for RLS
└── base_d6.pt           # Model checkpoint (if saved)
```

## Cost-Benefit Analysis

| Test Type | Time | Cost | Signal Quality |
|-----------|------|------|----------------|
| Quick local (100 steps) | 15min | Free | Sanity check |
| Standard local (200 steps) | 30-60min | Free | Basic validation |
| Thorough local (1000 steps) | 2-4hr | Free | Good signal |
| Overnight local (2000 steps) | 6-8hr | Free | Strong signal |
| Single GPU cloud | 1-2hr | $2-6 | Very strong |
| Full speedrun 8xH100 | 4hr | $100 | Production |

## Recommended Path

1. **Quick test:** Run default script (30-60 min)
2. **If successful:** Run overnight with 1000-2000 iterations
3. **If still good:** Single GPU cloud test
4. **If validated:** Full 8xH100 speedrun

This minimizes risk and cost while providing validation at each step.

## Questions?

Check the logs:
```bash
cd local_rls_experiments
less baseline.log
less rls.log
```

Compare training curves:
```bash
grep "train loss" baseline.log > baseline_losses.txt
grep "train loss" rls.log > rls_losses.txt
```

The goal: **Verify the code works correctly**, not to prove RLS is better (that requires scale).
