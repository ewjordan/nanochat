# Recurrent Layer State Implementation

## Overview

This document tracks the implementation of a recurrent layer state mechanism in nanochat. The feature enables the model to pass final layer activations from the previous token as additional input to the current token, potentially improving information flow and reducing redundant computation in lower layers.

## Motivation

In standard transformers, each layer at position N must rebuild contextual understanding from scratch using only:
- Token N's embedding
- Attention to positions 1..N-1's representations at the same layer

This mechanism adds a "side channel" where layer 1 processing token N can access the fully-processed representation of token N-1 from layer 20, potentially allowing:
- **Reduced cold start:** Lower layers get immediate access to high-level context
- **Better compute allocation:** Layers can specialize in integration vs. recomputation
- **Improved information flow:** Deep insights from previous tokens inform early processing

## Implementation Details

### Architecture Changes

**Config Parameters (`GPTConfig`):**
- `recurrent_layer_state: bool = False` - Enable/disable the feature
- `num_recurrence_warmup: int = 1` - Number of warmup passes for training

**New Module (`state_gate`):**
- `Linear(2 * n_embd, n_embd, bias=False)`
- Learns to mix concatenated `[prev_activation, token_embedding]` into combined representation
- Only created when `recurrent_layer_state=True`
- Initialized with zero weights (pass-through behavior initially)
- Added to Muon optimizer with other matrix parameters

**Forward Pass Modifications:**
- Added `prev_state` parameter (B, T, n_embd) - previous token's final layer activations
- Added `return_state` parameter - whether to return final layer activations
- Mixing happens before first norm: `cat([embedding, prev_state]) → state_gate → norm`
- Position 0 always uses zeros for prev_state (no previous token)
- Returns `(loss, final_state)` or `(logits, final_state)` when `return_state=True`

**Training Method (`forward_with_recurrence`):**
```python
def forward_with_recurrence(idx, targets):
    1. Initialize prev_state with zeros (B, T, n_embd)
    2. For _ in range(num_recurrence_warmup):
        a. No-grad forward pass to get activations
        b. Shift activations: position i gets position i-1's output
        c. Update prev_state
    3. Real forward pass with gradients using final prev_state
    4. Return loss
```

### Training Integration

**Training Loop Changes (`scripts/base_train.py`):**
- Added hyperparameters to global config
- Modified training loop to use `orig_model.forward_with_recurrence()` when enabled
- Falls back to normal `model(x, y)` when disabled

**Computational Cost:**
- `num_recurrence_warmup=1`: 2x forward compute per training step
- `num_recurrence_warmup=2`: 3x forward compute per training step
- Still only 1 backward pass (warmup passes are no-grad)

### Inference Integration

**Engine Changes (`nanochat/engine.py`):**
- During prompt prefill: Runs warmup passes to get initial states
- During decode: Tracks and passes previous token's real final state
- Handles state replication when expanding from batch 1 to multiple samples

**GPT.generate() Changes:**
- Warmup passes on initial prompt (consistent with training)
- Token-by-token generation passes real previous state
- No train/inference mismatch during actual generation

## Design Decisions

### 1. Gating Mechanism
**Chosen:** Concatenate both inputs and learn mixing via linear layer
- Flexibility: Can learn arbitrary combinations
- Sees both: Not forced into simple weighted average
- Future: Could add dropout for robustness

### 2. Initial State (Position 0)
**Chosen:** Zeros
- Simple and standard for recurrent mechanisms
- Considered learnable parameter but chose simplicity

### 3. Mixing Location
**Chosen:** Before first norm
- Mix → Norm creates consistent input to transformer blocks
- Avoids scale mismatch issues

### 4. Warmup Convergence
**Chosen:** Configurable number of passes
- Each pass "locks in" one more token from sequence start
- Convergence behavior to be studied empirically
- Default of 1 pass balances cost vs. benefit

## Known Limitations

### 1. Gradient Flow
Previous activations come from no-grad warmup passes, so:
- Model learns to USE prev_state (via state_gate weights)
- Model cannot learn to PRODUCE better activations for next token directly
- Signal is indirect through sequence-level loss

### 2. Train/Inference Mismatch (Training Only)
During warmup passes, model produces activation A₁ with stale prev_state, but trains on A₁.
During inference, we use actual previous activations.
Gap may limit effectiveness but is unavoidable with batched training.

### 3. Prompt Processing Latency
Initial prompt requires warmup passes, adding latency to every inference request.
Trade-off: Consistency with training vs. inference speed.

### 4. Computational Cost
2-4x forward compute overhead may not be justified if sample efficiency doesn't improve enough.
Question: Can smaller models + RLS outperform larger models without RLS for same compute?

## Testing & Validation

**Basic Functionality Tests:** ✅ All Passed
- Normal model creation and forward pass
- Recurrent model creation with state_gate
- forward_with_recurrence() warmup logic
- Loss computation in both modes

**Code Changes:**
- `nanochat/gpt.py`: +89 lines
- `nanochat/engine.py`: +33 lines
- `scripts/base_train.py`: +9 lines
- Total: ~131 lines added

**Backward Compatibility:**
When `recurrent_layer_state=False` (default), behavior is identical to original implementation.

## Usage

### Training with Recurrent Layer State

```bash
# Single GPU
python -m scripts.base_train -- --recurrent_layer_state=True --num_recurrence_warmup=1

# Distributed (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --recurrent_layer_state=True \
    --num_recurrence_warmup=2 \
    --depth=12
```

### Inference

Once a model is trained with RLS, inference automatically uses the mechanism (no flags needed).

## Experimental Questions

### Primary Questions
1. **Does RLS improve sample efficiency?** Does a d12 model with RLS match d16 without RLS?
2. **What's the optimal num_recurrence_warmup?** Do activations converge? How many passes needed?
3. **Which tasks benefit most?** Long sequences? Reasoning tasks? Perplexity?

### Secondary Questions
4. Does state_gate learn meaningful mixing, or does it learn gate→0 (ignore prev_state)?
5. How much does train/inference mismatch hurt performance?
6. Can dropout on state_gate improve robustness?
7. Does RLS help more with smaller models vs. larger models?

## Recommended Next Steps

### Phase 1: Proof of Concept
1. Train two d12 models (with/without RLS) on limited data (~1B tokens)
2. Evaluate perplexity and CORE metrics
3. Measure wall-clock time and compute cost
4. Check if RLS shows any positive signal

### Phase 2: Ablations (if Phase 1 promising)
1. Vary num_recurrence_warmup: [1, 2, 3, 5]
2. Try different model sizes: [d8, d12, d16, d20]
3. Measure activation convergence during warmup
4. Analyze state_gate learned weights

### Phase 3: Optimization (if Phase 2 validates approach)
1. Experiment with dropout on state_gate
2. Try different gating architectures
3. Investigate learned vs. fixed initial state
4. Optimize warmup pass efficiency

## References

This is a novel architectural experiment combining ideas from:
- **RNNs/LSTMs:** Passing hidden state between timesteps
- **State Space Models (Mamba, S4):** Recurrent state with transformers
- **Residual connections:** Providing direct paths for information flow

## Changelog

**2025-11-07:** Initial implementation
- Core architecture in gpt.py
- Training integration in base_train.py
- Inference support in engine.py and GPT.generate()
- All basic tests passing
- Feature complete and ready for experimentation
