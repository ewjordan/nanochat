# RLS Failure Analysis: Three Theories

## Executive Summary

This document investigates three theories for why Recurrent Layer State (RLS) training shows catastrophic failure at scale (validation bpb 1.76 vs baseline 1.13) but modest gaps locally (2.23 vs 2.16).

**Key Finding**: Local 500-step tests show gradient patterns are inconsistent and noisy, not systematically weaker in RLS. However, H100 runs at step 800 showed RLS has ~40% weaker gradients (1.69 vs 2.83). This suggests the problem emerges at scale.

---

## Architecture Overview

### Dual-Stream Attention (Layer 0 Only)

```
Input:
  - x: token embeddings (B, T, 768)
  - prev_state: hidden states from previous tokens (B, T, 768)

Layer 0 Processing:
  1. h_main = x + E_type_main
  2. h_side = prev_state + E_type_side

  3. q = c_q(h_main)           # queries from main stream only
  4. k_main = c_k(h_main)      # keys from main stream
  5. v_main = c_v(h_main)      # values from main stream
  6. k_side = c_k(h_side)      # keys from side stream
  7. v_side = c_v(h_side)      # values from side stream

  8. K = concat([k_main, k_side], dim=seq)  # (B, H, 2T, D)
  9. V = concat([v_main, v_side], dim=seq)  # (B, H, 2T, D)

  10. Attention: Q @ K^T with causal mask for both streams

Output: Attention output becomes input to subsequent layers
```

**Key Properties**:
- Only layer 0 has dual-stream attention
- Queries come from main stream only (side tokens never query)
- Both main and side K/V are attended to causally
- Type embeddings are the ONLY way to distinguish streams

---

## Theory 1: Side Token Attention Dominance

### Hypothesis
prev_state is a rich semantic representation that causes the model to copy information from side tokens instead of processing real input tokens, leading to rapid early learning (shortcuts) then plateau.

### Evidence FOR

1. **Side tokens are informationally richer**: `prev_state` comes from the final layer output of previous tokens - it's already been processed through 12 transformer layers. Main tokens are just raw embeddings.

2. **Attention pattern concern**: Queries can attend to both streams equally. If side K/V provide easier-to-extract information, attention might preferentially focus there.

3. **Early learning then plateau**: The H100 catastrophic runs showed RLS initially learned quickly but then plateaued around loss 6.0 while baseline continued improving to 4.5. This matches the "shortcut then stagnation" pattern.

4. **Type embeddings may be insufficient**: Only `E_type_main` and `E_type_side` (two 768-dim vectors) distinguish the streams. If these don't provide strong enough signal, the model may not learn to properly weight main vs side attention.

### Evidence AGAINST

1. **Causal masking prevents cheating**: Token i can only attend to positions j ≤ i in both main and side streams. It cannot attend to future information, so there's no direct "shortcut" to the answer.

2. **Side dropout (15%) maintains base competence**: During training, prev_state is randomly zeroed 15% of the time, forcing the model to maintain ability to process without side tokens.

3. **Queries come from main stream only**: Since `q = c_q(h_main)`, the attention mechanism is fundamentally driven by the main token representation, not side tokens.

4. **Local tests don't show catastrophic failure**: If side dominance were fundamental, we'd expect to see it at all scales, but 500-step local tests show only 3.6% gap.

### Testable Predictions

1. **Attention pattern logging**: If this theory is correct, we should see attention weights concentrate more on side positions (T:2T) than main positions (0:T) as training progresses.

2. **Scaling down side contribution**: Adding `h_side = alpha * prev_state + E_type_side` with alpha < 1.0 should improve performance if side tokens dominate.

3. **Gradient flow to side vs main**: If side tokens dominate, gradients to E_type_side should be much larger than to E_type_main.

### Tests to Run

```python
# 1. Log attention weights in layer 0
attn_weights = torch.nn.functional.softmax(q @ k.transpose(-2, -1) / sqrt(d), dim=-1)
main_attn = attn_weights[:, :, :, :T].mean()
side_attn = attn_weights[:, :, :, T:].mean()
print(f"Main attention: {main_attn:.4f}, Side attention: {side_attn:.4f}")

# 2. Scale down side contribution
h_side = 0.1 * prev_state + E_type_side  # Test alpha = 0.1

# 3. Compare type embedding gradients
print(f"E_type_main grad norm: {model.E_type_main.grad.norm():.6f}")
print(f"E_type_side grad norm: {model.E_type_side.grad.norm():.6f}")
```

---

## Theory 2: Batch Size Scaling Problem

### Hypothesis
Large batches (65,536) have less gradient noise and cause RLS to get stuck in local minima. The prev_state recurrence creates optimization landscape issues that scale poorly with batch size.

### Evidence FOR

1. **Local vs H100 batch size difference**:
   - Local 500-step tests: batch_size = 512 → gap = 3.6%
   - H100 catastrophic run: batch_size = 65,536 → gap = 56%
   - This is a 128x difference in batch size

2. **Gradient noise helps escape local minima**: Smaller batches provide noisy gradient estimates that can help escape shallow local minima. RLS's recurrent structure might create a more rugged loss landscape.

3. **Current H100 test with batch=512 is in progress**: This directly tests the hypothesis. If batch size is the cause, we should see RLS perform much better with batch=512 on H100.

4. **Learning rate scaling**: The learning rate isn't scaled with batch size in the current setup, which could cause optimization issues at large batch sizes.

### Evidence AGAINST

1. **Baseline handles large batches fine**: The baseline model trains perfectly well with batch_size=65,536, suggesting the batch size itself isn't inherently problematic.

2. **Early gradient weakness**: H100 run at step 800 showed RLS with weaker gradients (1.69 vs 2.83) even very early in training. If this were purely a local minimum issue, we'd expect similar gradients initially.

3. **Gradient accumulation should be equivalent**: Whether you do 128 steps of batch=512 or 1 step of batch=65,536, the gradient should be mathematically similar (ignoring batch normalization effects, which we don't use).

4. **No momentum in optimizers**: Both Muon and AdamW use momentum, which should help smooth out optimization regardless of batch size.

### Testable Predictions

1. **H100 batch=512 run should succeed**: If batch size is the root cause, the ongoing H100 run with batch=512 should show RLS achieving validation bpb close to baseline.

2. **Gradient noise metric**: Measuring gradient variance across mini-batches should show RLS has higher variance than baseline, suggesting it benefits more from noise.

3. **Loss landscape visualization**: 2D loss landscape plots (varying two model parameters) should show RLS has more local minima than baseline.

### Tests to Run

```python
# 1. Gradient variance tracking
gradients_buffer = []
for step in range(10):
    loss = forward_pass()
    loss.backward()
    grad_snapshot = torch.cat([p.grad.flatten() for p in model.parameters()])
    gradients_buffer.append(grad_snapshot)
grad_variance = torch.stack(gradients_buffer).var(dim=0).mean()

# 2. Try intermediate batch sizes
# Test: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
# Find the threshold where RLS starts failing

# 3. Learning rate scaling experiment
# Scale LR proportionally to sqrt(batch_size)
# lr_scaled = lr_base * sqrt(batch_size / 512)
```

---

## Theory 3: Gradient Flow Architecture Issue

### Hypothesis
The dual-stream attention and prev_state dependency creates a recurrent structure that fundamentally dampens gradients through the main pathway, causing the model to learn slower regardless of batch size.

### Evidence FOR

1. **H100 early gradient weakness**: At step 800, RLS showed 1.69 grad norm vs baseline 2.83 grad norm. This is a ~40% reduction VERY early in training, suggesting a fundamental architectural issue.

2. **Doubled sequence length in layer 0**: Layer 0 attention operates on 2T tokens instead of T. This means:
   - Attention scores are computed over 2x positions
   - Softmax denominator includes 2T terms (dilutes gradients)
   - Gradients flow through longer paths

3. **Information bypass hypothesis**: prev_state already contains processed information from previous tokens. The model might learn to rely on this "easy path" instead of building strong representations from scratch, leading to weaker gradients on the main pathway.

4. **Warmup cost doubles computation**: Each training step requires a warmup pass with `torch.no_grad()` to compute prev_state for all tokens. This doubles the forward pass cost, which might indicate fundamental inefficiency.

5. **Type embeddings receive different gradient magnitudes**: Previous diagnostics (from summary) showed E_type_side was 70x larger than E_type_main in magnitude, suggesting asymmetric learning.

### Evidence AGAINST

1. **Local tests show only modest gap**: If gradient flow were fundamentally broken, we'd expect catastrophic failure at all scales. The 500-step local test showed only 3.6% gap (2.23 vs 2.16 bpb).

2. **Gradient norm comparison is noisy**: Looking at the 500-step logs step-by-step:
   - Step 50: RLS 6.58 > Baseline 4.48 (+47% STRONGER)
   - Step 100: RLS 4.84 > Baseline 3.80 (+27% stronger)
   - Step 250: RLS 1.86 < Baseline 5.70 (-67% weaker)
   - Step 400: RLS 1.43 < Baseline 7.40 (-81% weaker)
   - Step 450: RLS 0.88 < Baseline 1.53 (-42% weaker)

   The gradient pattern is inconsistent - sometimes RLS has stronger gradients, sometimes weaker. This doesn't support a systematic gradient dampening theory.

3. **Final loss at step 450 is similar**: RLS 7.26 vs Baseline 7.31. Despite the gradient fluctuations, the training loss is nearly identical.

4. **Residual connections bypass layer 0**: Even if layer 0 has weak gradients, residual connections allow gradients to flow directly to embeddings, bypassing the problematic dual-stream attention.

### Testable Predictions

1. **Layer-wise gradient analysis**: If this theory is correct, layer 0 should have consistently weaker gradients than other layers in RLS, but not in baseline.

2. **Gradient flow to embeddings**: Gradients to `wte` (token embeddings) should be weaker in RLS than baseline, indicating the main pathway is learning slower.

3. **Attention output magnitude**: If the information bypass is happening, the attention output from layer 0 should have lower magnitude in RLS (model is relying on residual connection).

4. **Removing dual-stream increases gradients**: A version that just adds prev_state to layer 0 input without dual-stream attention should have stronger gradients.

### Tests to Run

```python
# 1. Layer-wise gradient norms
for i, block in enumerate(model.transformer.h):
    attn_grad = torch.cat([p.grad.flatten() for p in block.attn.parameters()]).norm()
    mlp_grad = torch.cat([p.grad.flatten() for p in block.mlp.parameters()]).norm()
    print(f"Layer {i}: attn_grad={attn_grad:.4f}, mlp_grad={mlp_grad:.4f}")

# 2. Embedding gradient comparison
wte_grad_norm = model.transformer.wte.weight.grad.norm()
print(f"wte grad norm: {wte_grad_norm:.6f}")

# 3. Attention output magnitude
# During forward pass, log:
attn_output = self.attn(norm(x), ...)
print(f"Layer {i} attn output norm: {attn_output.norm():.4f}")

# 4. Test simplified prev_state integration
# Replace dual-stream with:
# x = x + 0.1 * prev_state  # Direct addition to layer 0 input
```

---

## Gradient Data Analysis: Local 500-Step Tests

### Full Gradient Comparison Table

| Step | RLS Loss | RLS Grad | Base Loss | Base Grad | Grad Δ   | Loss Δ   |
|------|----------|----------|-----------|-----------|----------|----------|
| 0    | 11.090   | 1.415    | 11.090    | 1.415     | 0.000    | 0.000    |
| 50   | 8.055    | 6.578    | 8.025     | 4.477     | +2.101   | +0.030   |
| 100  | 7.868    | 4.839    | 7.882     | 3.797     | +1.042   | -0.014   |
| 150  | 7.385    | 4.081    | 7.301     | 3.152     | +0.929   | +0.084   |
| 200  | 7.839    | 4.725    | 7.630     | 2.581     | +2.144   | +0.209   |
| 250  | 7.551    | 1.860    | 7.467     | 5.698     | -3.838   | +0.084   |
| 300  | 7.239    | 1.969    | 7.134     | 1.572     | +0.397   | +0.105   |
| 350  | 7.401    | 1.973    | 7.370     | 1.712     | +0.261   | +0.031   |
| 400  | 7.441    | 1.434    | 7.458     | 7.397     | -5.963   | -0.017   |
| 450  | 7.258    | 0.882    | 7.306     | 1.527     | -0.645   | -0.048   |
| 500  | Val:2.233| -        | Val:2.155 | -         | -        | +0.078   |

### Key Observations

1. **Gradient noise is extreme**: Gradient norms vary wildly for both models (baseline ranges from 1.5 to 7.4).

2. **No systematic pattern**: RLS gradients are not consistently weaker. In 6 out of 9 steps, RLS has stronger gradients than baseline.

3. **Final validation gap is small**: 2.233 vs 2.155 bpb = 3.6% gap, which is much better than the 56% gap seen on H100 at scale.

4. **Training losses converge**: At step 450, RLS loss (7.258) is actually slightly better than baseline (7.306).

5. **Gradient dampening at end**: Steps 250, 400, and 450 show RLS with weaker gradients as loss approaches convergence. This might be normal (loss is flatter near minimum).

### Interpretation

The local gradient data does NOT strongly support Theory 3 (systematic gradient dampening). The gradients are noisy for both models, and RLS often has stronger gradients than baseline.

However, this conflicts with H100 step 800 data showing systematic gradient weakness. This suggests:
- **Scale-dependent effect**: The gradient problem emerges at larger scale/longer training
- **Batch size interaction**: Large batches might amplify a latent gradient issue
- **Early vs late training**: The problem develops over time as the model learns to exploit shortcuts

---

## H100 Data Analysis (From Summary)

### Catastrophic Run (batch_size=65,536, with side_mlp)

```
Step 500:
- RLS: loss 6.14, validation bpb 1.7609
- Baseline: loss 4.53 (estimated), validation bpb 1.1300
- Gap: 56% worse
```

**Pattern**: Both old run (with side_mlp) and new run (without side_mlp) showed identical trajectories:
- Initial rapid learning to ~6.0 loss
- Plateau around 6.0 loss
- Never improved beyond that point
- Baseline continued improving to ~4.5 loss

### New Run Without side_mlp (batch_size=65,536)

```
Step 500:
- RLS: loss 6.14, grad norm ~0.17 (estimated from trend)
- Baseline: not available (different run)
```

WandB comparison showed **identical curves** for RLS with and without side_mlp, strongly suggesting side_mlp wasn't the root cause.

### Step 800 Observation (Current batch_size=512 test on H100)

```
Step 800:
- RLS: loss 7.89, grad norm 1.69
- Baseline: loss 7.15, grad norm 2.83
- Gradient gap: -40%
- Loss gap: +10%
```

**This is the smoking gun**: Very early in training (0.2% complete), RLS already shows:
- Weaker gradients (1.69 vs 2.83)
- Higher loss (7.89 vs 7.15)
- Slower learning rate

This suggests a fundamental architectural issue, not just batch size.

---

## Synthesis: Which Theory Best Explains the Data?

### Theory 1 (Side Token Attention Dominance): **PLAUSIBLE**

**Fits the data:**
- Explains early learning then plateau (model finds shortcut via side tokens)
- Explains why removing side_mlp didn't help (problem is in attention pattern, not pre-attention processing)
- E_type_side being 70x larger than E_type_main suggests asymmetric learning

**Doesn't fit:**
- Local tests show only modest gap
- Causal masking prevents direct cheating
- Side dropout should prevent over-reliance

**Verdict**: Requires attention pattern analysis to confirm/refute.

### Theory 2 (Batch Size Scaling): **PARTIALLY SUPPORTED**

**Fits the data:**
- Clear correlation between batch size and performance (512 → 3.6% gap, 65,536 → 56% gap)
- Currently being tested with H100 batch=512 run
- Would explain scale-dependent failure

**Doesn't fit:**
- H100 step 800 shows gradient weakness even with batch=512
- Baseline handles large batches fine
- No theoretical reason why RLS should be more sensitive to batch size

**Verdict**: Batch size affects severity, but likely not the root cause. The H100 batch=512 run showing early gradient weakness suggests batch size amplifies an underlying issue rather than causing it.

### Theory 3 (Gradient Flow Architecture): **STRONGLY SUPPORTED**

**Fits the data:**
- H100 step 800 shows systematic gradient weakness (1.69 vs 2.83) very early
- Dual-stream attention dilutes gradients over 2T positions
- E_type_side being 70x larger suggests asymmetric gradient flow
- Explains why removing side_mlp didn't help (core issue is in dual-stream attention)

**Doesn't fit:**
- Local 500-step gradient data shows inconsistent pattern
- Final local training losses are similar
- Residual connections should bypass gradient issues

**Verdict**: Most likely root cause, especially given H100 early gradient data. The local gradient inconsistency might be due to:
- Gradient noise overwhelming the signal at small scale
- Different device (MPS vs CUDA) affecting numerical behavior
- Need for longer training to see the pattern emerge

---

## Recommended Next Steps

### 1. Immediate: Wait for H100 batch=512 run to complete

This will test Theory 2 directly. If RLS still fails catastrophically with batch=512, we can rule out batch size as the primary cause.

### 2. High Priority: Attention Pattern Analysis

Add logging to measure:
```python
# In layer 0 forward, after computing attention
attn_scores = (q @ k.transpose(-2, -1)) / sqrt(d)  # Before softmax
main_scores = attn_scores[:, :, :, :T].mean()
side_scores = attn_scores[:, :, :, T:].mean()
print(f"Step {step}: main_attn_score={main_scores:.4f}, side_attn_score={side_scores:.4f}")
```

This tests Theory 1. If side_scores >> main_scores, we have a smoking gun.

### 3. High Priority: Layer-wise Gradient Analysis

```python
# After backward pass, before optimizer step
for i, block in enumerate(model.transformer.h):
    attn_params = list(block.attn.parameters())
    attn_grad = torch.stack([p.grad.norm() for p in attn_params if p.grad is not None]).mean()
    print(f"Layer {i} attn grad: {attn_grad:.6f}")
```

This tests Theory 3. If layer 0 has systematically weaker gradients, we confirm gradient flow issue.

### 4. Medium Priority: Type Embedding Gradient Tracking

```python
# Log every N steps
if step % 100 == 0:
    print(f"E_type_main grad: {model.E_type_main.grad.norm():.6f}")
    print(f"E_type_side grad: {model.E_type_side.grad.norm():.6f}")
    print(f"Ratio: {model.E_type_side.grad.norm() / model.E_type_main.grad.norm():.2f}x")
```

This helps diagnose asymmetric learning in both Theory 1 and Theory 3.

### 5. Architectural Experiments (If gradient flow confirmed)

If Theory 3 is confirmed, try:

**A. Scale down prev_state contribution:**
```python
h_side = 0.1 * prev_state + E_type_side
```

**B. Remove dual-stream attention (direct integration):**
```python
# In layer 0, instead of dual-stream:
x = x + 0.1 * prev_state  # Direct additive integration
```

**C. Apply prev_state at higher layer:**
```python
# Try layer 1 or layer 2 instead of layer 0
# Hypothesis: Later layers have stronger gradients from deeper computation
```

**D. Use prev_state only as residual (not in attention):**
```python
# Layer 0 normal attention
attn_output = self.attn(x, ...)
# Add prev_state after attention
attn_output = attn_output + 0.1 * prev_state
```

---

## Conclusion

**Most likely explanation**: Theory 3 (Gradient Flow Architecture Issue) is the root cause, with Theory 2 (Batch Size) amplifying the problem.

**Evidence**:
1. H100 step 800 shows systematic gradient weakness (1.69 vs 2.83) even with batch=512
2. Dual-stream attention dilutes gradients over 2T positions
3. Removing side_mlp didn't help, suggesting issue is in attention mechanism itself
4. E_type_side dominance (70x) suggests asymmetric gradient flow

**Next action**: Add layer-wise gradient logging and attention pattern analysis to confirm the root cause, then test architectural modifications to fix gradient flow.
