# Local Validation Guide for Recurrent Layer State

This guide helps you validate the recurrent layer state (RLS) feature locally on your M4 Mac before running expensive GPU training.

## Quick Start

### 1. Download minimal training data (optional but recommended)

```bash
# Download just 1 shard (~100MB) for testing
python -m nanochat.dataset -n 1
```

### 2. Run the validation script

```bash
python local_validation_run.py
```

This will:
- Train two small models (depth=6, ~7M params each)
- One baseline (no RLS)
- One with RLS enabled
- Compare final losses after 100 steps
- Should complete in 5-15 minutes on M4 Mac

## What to Look For

### ✅ Success Criteria

1. **No crashes**: Both models train without errors
2. **Decreasing loss**: Loss should go down for both models
3. **state_gate learns**: RLS model shows the new mechanism is active
4. **Reasonable loss**: Final loss should be in a sane range (3-5 for tiny test)

### 📊 Performance Comparison

The 100-step test is **too short** to show real performance differences. You might see:
- RLS slightly better: Good sign, but not conclusive
- RLS slightly worse: Normal, warmup overhead matters more in short runs
- About the same: Expected for such a tiny test

**Real validation requires:**
- 1000+ steps minimum
- Proper eval metrics (not just train loss)
- Multiple runs for statistical significance

## Customization

Edit `local_validation_run.py` to adjust:

```python
depth = 6          # Model size (6 = 7M params, 8 = 13M params, 10 = 20M params)
num_steps = 100    # Training steps (100 = quick test, 500 = better signal, 1000 = real test)
```

## Next Steps

If the local validation succeeds:

### Option 1: Longer Local Run (recommended)
```bash
# Edit local_validation_run.py to set:
#   num_steps = 1000
#   depth = 8 or 10
python local_validation_run.py
```

This will take 1-2 hours but give much better signal.

### Option 2: Single GPU Cloud Test
Before committing to 8xH100, test on a single GPU:
```bash
# On a single H100/A100
python -m scripts.base_train -- \
    --depth=12 \
    --num_iterations=1000 \
    --recurrent_layer_state=True \
    --num_recurrence_warmup=1
```

### Option 3: Full Speedrun (when confident)
```bash
# 8xH100 node (~$100, 4 hours)
WANDB_RUN=rls_test bash speedrun.sh
```

## Troubleshooting

**"No module named torch"**
```bash
source .venv/bin/activate
python local_validation_run.py
```

**"No training data found"**
- The script will work with random data (for basic functionality testing)
- For better results, download data: `python -m nanochat.dataset -n 1`

**"Out of memory"**
- Reduce depth: Try `depth = 4`
- The script uses very small batches (2) and sequences (512) to fit on Mac

**Very slow on MPS**
- MPS can be slower than expected for small batches
- This is normal, just a validation test
- GPU training will be much faster

## Understanding the Output

```
TEST 1: Baseline (no recurrent layer state)
  Model parameters: 7,123,456
  Step 20/100 | Loss: 4.5234 | 150.2ms/step
  ...
  Final loss (avg last 10): 4.1234

TEST 2: With recurrent layer state
  Model parameters: 7,131,648  # Slightly more (state_gate params)
  Step 20/100 | Loss: 4.5189 | 310.4ms/step  # ~2x slower (warmup passes)
  ...
  Final loss (avg last 10): 4.1156

RESULTS
Baseline final loss:  4.1234
RLS final loss:       4.1156
✅ RLS improved by 0.2%
```

**Key observations:**
- RLS adds ~8K parameters (minimal)
- RLS is ~2x slower (expected: warmup passes)
- Small improvements in short runs are **not** conclusive
- The goal is to verify it works, not to prove it's better

## Files Created

- `local_validation_run.py` - Main validation script
- `test_state_gate_init.py` - Unit tests for initialization
- `test_gradient_simple.py` - Gradient flow tests
- `test_bfloat16_issue.py` - Device compatibility tests
- `test_training_steps.py` - Multi-step learning test

You can delete these test files after validation if desired.
