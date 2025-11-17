# State Gate Initialization Fix - Summary

## The Problem Found

The `state_gate` was initialized with **all zeros**, which caused:

```python
# Old initialization (BROKEN)
torch.nn.init.zeros_(self.state_gate.weight)

# What happened:
gate_input = [token_embedding, prev_state]  # Concatenate inputs
output = zeros_matrix @ gate_input = ZEROS  # All zeros!
# Token embeddings completely destroyed! 🔥
```

The model would receive all-zero inputs to the transformer blocks, making it completely non-functional.

## The Fix

New initialization that **actually** provides pass-through behavior:

```python
# New initialization (CORRECT)
# First half: Identity matrix (pass through token embeddings)
self.state_gate.weight[:, :n_embd] = torch.eye(n_embd)
# Second half: Zeros (ignore prev_state initially)
self.state_gate.weight[:, n_embd:] = 0.0

# Now:
output = I @ token_emb + 0 @ prev_state = token_emb  # Works! ✅
```

## Verification

Tested on your M4 Mac (MPS device):

- ✅ Token embeddings pass through correctly
- ✅ Model trains without errors
- ✅ Gradients flow properly (after warmup step)
- ✅ state_gate learns over training (gradients increase from 0.000 → 0.002)
- ✅ Both with/without RLS modes work

## Next Steps: Local Validation

Before spending $100 on 8xH100 training:

### Option 1: Quick Test (5-10 minutes)
```bash
source .venv/bin/activate
python local_validation_run.py
```

This runs a tiny test (depth=6, 100 steps) to verify:
- No crashes
- Basic functionality
- Loss decreases

### Option 2: Better Test (1-2 hours) - **RECOMMENDED**
```bash
# Download 1-2 data shards first
python -m nanochat.dataset -n 2

# Edit local_validation_run.py:
#   depth = 8 or 10
#   num_steps = 1000

python local_validation_run.py
```

This gives better signal about whether RLS helps.

### Option 3: Single GPU Cloud Test
Before committing to 8xH100, test on single H100/A100:
```bash
python -m scripts.base_train -- \
    --depth=12 \
    --num_iterations=2000 \
    --recurrent_layer_state=True
```

Compare to baseline:
```bash
python -m scripts.base_train -- \
    --depth=12 \
    --num_iterations=2000 \
    --recurrent_layer_state=False
```

### Option 4: Full Speedrun (when confident)
```bash
WANDB_RUN=rls_experiment bash speedrun.sh
```

## Git Status

```bash
# Committed fix:
git log --oneline -1
# 0ff4976 Fix state_gate initialization to prevent zeroing token embeddings

# Current branch:
git branch
# * recurrent-layer-state

# Ready to push or test
```

## Files Created

**Main fix:**
- `nanochat/gpt.py` - Fixed initialization

**Validation tools:**
- `local_validation_run.py` - Local testing script
- `LOCAL_VALIDATION.md` - Detailed guide

**Test files (can delete after validation):**
- `test_state_gate_init.py`
- `test_gradient_simple.py`
- `test_bfloat16_issue.py`
- `test_training_steps.py`

## Expected Behavior

**On first training step:**
- Most gradients near zero (due to nanochat's zero-init strategy)
- Only lm_head has significant gradients
- This is **normal and expected**

**After a few steps:**
- state_gate gradients appear and grow
- Embedding gradients appear
- Model learns normally

**Performance:**
- RLS is ~2x slower (warmup overhead)
- Whether it improves sample efficiency is TBD (requires real training)
- Small local tests won't show meaningful performance differences

## Recommendation

1. **Run Option 2** (1-2 hour local test with real data)
2. If that looks good, **run Option 3** (single GPU cloud test)
3. If both look promising, **run full speedrun** on 8xH100

This de-risks the expensive training run.
