# torch.compile None Gradient Issue

## Summary

When using `torch.compile` with RLS (Recurrent Layer State) training, some parameters receive `None` gradients on the first training step, causing optimizer crashes. This appears to be related to torch.compile creating different compiled kernels for different forward pass code paths.

## Environment

- PyTorch version: 2.x (with torch.compile)
- GPU: NVIDIA A100 40GB
- Model: GPT-style transformer (186M params, 12 layers)
- Training: Mixed precision (bfloat16)

## Issue Details

### What Happens

1. Model uses two forward pass variants:
   - **Warmup pass**: `forward(idx, targets=None, prev_state=prev_state, return_state=True)`
     - Skips lm_head computation (optimization)
     - Returns hidden states only
   - **Training pass**: `forward(idx, targets=targets, prev_state=prev_state, return_state=False)`
     - Computes full forward + loss

2. On the first training step:
   - Warmup runs in `torch.no_grad()` context
   - Training pass runs with gradients enabled
   - Some parameters get `None` gradients
   - Optimizers crash with `AssertionError: g is not None` (Muon) or AttributeError (AdamW)

### Code Location

**Model code** (`nanochat/gpt.py`):
```python
def forward_with_recurrence(self, idx, targets=None, loss_reduction='mean'):
    # Warmup pass (no gradients)
    for _ in range(self.config.num_recurrence_warmup):
        with torch.no_grad():
            _, warmup_state = self.forward(idx, targets=None, prev_state=prev_state, return_state=True)
            # ... state shifting logic ...

    # Real forward pass (with gradients)
    return self.forward(idx, targets=targets, loss_reduction=loss_reduction, prev_state=prev_state, return_state=False)
```

**Optimizer crash** (`nanochat/muon.py:75`):
```python
def step(self):
    for group in self.param_groups:
        for p in group["params"]:
            g = p.grad
            assert g is not None  # ❌ Fails here on first RLS training step
```

## Root Cause Hypothesis

`torch.compile` creates separate compiled kernels for:
1. The warmup forward pass (with `return_state=True`, no lm_head)
2. The training forward pass (with `targets` and full computation)

These different code paths may cause torch.compile to:
- Mark certain parameters as "unused" in one variant
- Not propagate gradients to those parameters
- Leave gradients as `None` on the first step

Likely affected parameters:
- `state_gate` (only used when `prev_state is not None`)
- Possibly other conditionally-used parameters

## Workaround

Modified optimizers to skip parameters with `None` gradients:

**Muon** (`nanochat/muon.py`):
```python
def step(self):
    for group in self.param_groups:
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue  # Skip parameters without gradients
            # ... rest of update logic ...
```

**DistAdamW** (`nanochat/adamw.py`):
```python
def step(self):
    # First loop: gather gradients
    for group in self.param_groups:
        for p in group["params"]:
            grad = p.grad
            if grad is None:
                continue  # Skip parameters without gradients
            # ... gradient reduction logic ...

    # Second loop: apply updates
    for group in self.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue  # Skip parameters without gradients
            # ... parameter update logic ...
```

## Reproduction Steps

1. Create a model with conditional parameters (e.g., `state_gate` used only when `prev_state is not None`)
2. Use `torch.compile` on the model
3. Create a training loop with:
   - Warmup forward pass: `with torch.no_grad(): forward(targets=None, return_state=True)`
   - Training forward pass: `forward(targets=targets, return_state=False)`
4. Run optimizer step on first training iteration
5. Observe `None` gradients for conditionally-used parameters

## Questions for PyTorch Team

1. Is this expected behavior when torch.compile sees different code paths?
2. Should optimizers always handle `None` gradients gracefully?
3. Is there a way to force torch.compile to use a single kernel for multiple forward variants?
4. Could `set_to_none=True` in `optimizer.zero_grad()` interact with this issue?

## Related Issues

- Potentially related to torch.compile's tracing behavior with conditional execution
- May be similar to issues with torch.compile + gradient checkpointing
- Could be related to ZeRO-style sharded optimizers with torch.compile

## Upstream Issue Filing

**Repository**: https://github.com/pytorch/pytorch
**Labels**: `module: compile`, `module: autograd`, `triaged`
**Priority**: Medium (workaround exists, but unexpected behavior)

## Status

- **Workaround implemented**: ✅ (commit: acc825f)
- **Upstream issue filed**: ❌ (TODO)
- **Root cause confirmed**: ❌ (hypothesis only)

---

**Last updated**: 2025-11-08
**Author**: Claude Code
**Related commits**:
- `acc825f` - Skip parameters with None gradients in optimizers
- `2a1e900` - Add memory profiling to RLS forward pass
- `6fea327` - Fix RLS OOM by skipping lm_head during warmup passes
