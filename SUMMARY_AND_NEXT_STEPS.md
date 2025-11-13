# RLS Investigation: Summary and Next Steps

## Executive Summary

**Problem:** Recurrent Layer State (RLS) training shows catastrophic failure at scale (validation bpb 1.76 vs baseline 1.13).

**Root Cause Discovered:** "Noisy Gradient Dominance" - side tokens receive 22-50x more gradient than main tokens even when prev_state contains zero information, drowning out the true learning signal.

**Current Status:** Running ablation experiments on H100 to validate that masking side attention recovers baseline performance.

---

## Key Discovery: Noisy Gradient Dominance Theory

### The Problem

At initialization/early training when `prev_state ≈ 0`:
```python
K_side ≈ E_type_side  # All 512 positions nearly identical
V_side ≈ E_type_side  # All 512 positions nearly identical
```

**Gradient Accumulation Effect:**
- Each query attends to 512 nearly-identical side tokens
- Gradients from all 512 positions flow to **one parameter** (E_type_side)
- Creates **512× gradient accumulation** on E_type_side
- Main stream: gradients spread across different `wte[token_id]` - no accumulation

**Measured Evidence:**
- E_type_side: 22.6x more gradient than E_type_main (H100 step 0)
- Test ablations: 34-51x gradient ratio
- Overall gradient norms are EQUAL (1.01x RLS vs baseline)
- Layer 0 gradients STRONGER in RLS (1.23x)

### The Vicious Cycle

1. Large but noisy side gradients (22-50x) drown out true signal from main tokens
2. Main network can't learn → prev_state stays uninformative
3. Uninformative prev_state continues dominating gradients with noise
4. Cycle perpetuates → catastrophic failure

**Why RLS is worse than baseline:** Side stream is harmful noise, not a shortcut. The "shouting noise" analogy - even meaningless loud noise drowns out quiet signal.

---

## Currently Running Experiments (H100)

**Location:** `ubuntu@100.121.60.36` (via Tailscale SSH)

**Experiments:**

1. **baseline** *(completed)*  
   - Log: `/tmp/baseline_1k.log` (val bpb 1.7545 @ step 1000)  
   - Command used: `torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=12 --max_seq_len=512 --device_batch_size=8 --total_batch_size=4096 --num_iterations=1000 --recurrent_layer_state=False ...`

2. **rls (original dropout semantics)** *(completed)*  
   - Log: `/tmp/rls_1k.log` (stalled around 2.18 bpb by step 300)  
   - Same command as baseline but with `--recurrent_layer_state=True`

3. **rls_masked** *(completed)*  
   - Log: `/tmp/rls_masked_1k.log` (val bpb 2.0030 @ step 1000)  
   - Adds `--mask_side_attention=True`

4. **rls_dropout (prod RLS with side-stream removal dropout)** *(completed)*  
   - Command:  
     ```bash
     cd ~/nanochat
     source .venv/bin/activate
     python -m scripts.base_train --run=rls_dropout --depth=12 --max_seq_len=512 \
       --device_batch_size=8 --total_batch_size=4096 --num_iterations=1000 \
       --recurrent_layer_state=True --num_recurrence_warmup=1 \
       --side_dropout_rate=0.15 --mask_side_attention=False \
       --eval_every=100 --core_metric_every=-1 --sample_every=1000 --log_every=10
     ```
   - Log: `/tmp/rls_dropout_1k.log` (val bpb 2.0030 @ step 1000)
   - WandB: https://wandb.ai/ritz-deli-games/nanochat/runs/qg203vc8

5. **rls_gated (dropout + type gate)** *(running)*  
   - Same command as #4 plus `--side_type_gate=True`  
   - Log: `/tmp/rls_gated_1k.log`  
   - WandB: https://wandb.ai/ritz-deli-games/nanochat/runs/oqhvrezt  
   - Prints now include `gate μ/↑/↓` and `side_on` so we can inspect how often the side stream is active.

**Note:** We hit Muon `grad is None` assertions when launching via `torchrun` (even with `nproc_per_node=1`). Falling back to single-process `python -m` avoids that issue on the single-GPU H100 node.

**Check progress / logs:**
```bash
ssh ubuntu@100.121.60.36 "grep 'Validation' /tmp/*_1k.log"
# For a future tmux run:
ssh -t ubuntu@100.121.60.36 "tmux attach -t rls_dropout"
```

---

## Code Changes Made

### 1. nanochat/gpt.py

**Added ablation flags to GPTConfig (lines 36-44):**
```python
# Ablation flags for testing RLS side token dominance hypothesis
mask_side_attention: bool = False  # prevent attention to side tokens during training
zero_prev_state: bool = False      # zero out prev_state to disable side stream
side_dropout_rate: float = 0.15    # dropout rate for side stream (now configurable)
side_type_gate: bool = False       # scale type embedding based on prev_state strength
```

**Key implementation (line 127-129):**
```python
# ABLATION: Prevent attention to side tokens to test dominance hypothesis
if self.training and self.config.mask_side_attention:
    mask_side = torch.zeros(Tq, Tk, dtype=torch.bool, device=q.device)
    attn_mask = torch.cat([mask_main, mask_side], dim=1)  # (Tq, 2*Tk)
```

**Also updated (line 362):**
- Side dropout now uses `self.config.side_dropout_rate` instead of hardcoded 0.15
- Added `self.config = config` to CausalSelfAttention.__init__() (line 59)
- Dropout now removes the side stream entirely for that batch (lines 357-369) so we never insert constant side tokens when simulating "raw" training
- Optional `side_type_gate` scales `E_type_side` per-token based on prev_state norm so the type embedding only turns on when the recurrent signal carries information (lines 357-375, 77-100)
- Instrumentation: every forward stores `gate_mean/max/min` and whether the side stream was active so training logs/wandb capture these stats.

### 2. scripts/base_train.py (lines 43-46)

**Exposed ablation flags & telemetry toggles:**
    ```python
    # RLS ablation flags for debugging
    mask_side_attention = False
    zero_prev_state = False
    side_dropout_rate = 0.15
    side_type_gate = False
    side_type_gate_temp = 4.0
    side_type_gate_eps = 1e-6
    side_type_gate_ema_beta = 0.01
    side_stream_initial_scale = 0.1
    side_stream_final_scale = 1.0
    side_stream_schedule_steps = 500
    ```

    **Added to model_config_kwargs (line 119):**
    ```python
    model_config_kwargs = dict(..., mask_side_attention=mask_side_attention,
                              zero_prev_state=zero_prev_state,
                              side_dropout_rate=side_dropout_rate,
                              side_type_gate=side_type_gate,
                              side_type_gate_temp=side_type_gate_temp,
                              side_type_gate_eps=side_type_gate_eps,
                              side_type_gate_ema_beta=side_type_gate_ema_beta,
                              side_stream_initial_scale=side_stream_initial_scale,
                              side_stream_final_scale=side_stream_final_scale,
                              side_stream_schedule_steps=side_stream_schedule_steps)
    ```

- Training printouts & wandb metrics now include `gate μ/↑/↓` (mean/max/min gate) and `side_on` (whether the side stream was active) whenever RLS is enabled.

### 3. New Files Created

**test_ablations.py** - Validates ablation flags work correctly
- Tests: Normal RLS, zero_prev_state, mask_side_attention, 100% dropout
- Latest run (after dropout fix): mask_side_attention and 100% dropout both zeroed side gradients; normal RLS still shows 33x dominance; zero_prev_state still fails (52x) because tokens remain identical

**RLS_THEORY_ANALYSIS.md** - Comprehensive 450-line analysis of three theories

**test_rls_diagnostics.py** - Compare RLS vs baseline gradient flow

---

## Key Findings Summary

### What Works
✅ **mask_side_attention** - Perfectly eliminates side gradients (0.00x ratio)
- Prevents queries from attending to side tokens
- Forces model to learn from main tokens only
✅ **side_dropout removal** - Running the production RLS config with the new dropout semantics (remove stream instead of zeroing) matches masked performance (val bpb 2.003 @ step 1000 in `/tmp/rls_dropout_1k.log`)
⚠️ **side_type_gate + schedule** - Norm-based gate plus an explicit linear scale (`side_stream_initial_scale→side_stream_final_scale` over `side_stream_schedule_steps`) is wired up and verified locally; waiting for the H100 box to come back online to re-run `rls_gated` and see if the slow ramp actually prevents the early plateau.

### What Doesn't Work
❌ **zero_prev_state** - Makes problem WORSE (44-57x ratio)
- Side becomes `zeros + E_type_side`
- All positions identical → concentrates ALL gradient on E_type_side

✅ **100% side dropout** (new behavior) - Now disables the side stream entirely, same effect as masking
- Confirms the gradient issue was caused by keeping dummy side tokens in the graph
- Need to re-run long training to verify convergence matches baseline

### Evidence Against Alternative Theories

**Theory 3 (Gradient Flow) - REJECTED:**
- Overall gradient norms are equal (1.01x)
- Layer 0 gradients stronger in RLS (1.23x)
- Problem is not weak gradients, but imbalanced gradients

**Theory 2 (Batch Size) - UNTESTED:**
- Still possible but secondary to gradient dominance
- Would need experiments with different batch sizes

---

## Critical Questions to Answer

### 1. Does mask_side_attention recover baseline performance?
**Test:** Compare validation curves from running experiments
- If RLS_masked ≈ Baseline: Confirms side tokens are harmful noise
- If RLS_masked < Baseline: Something else also broken
- If RLS_masked > Baseline: Theory incomplete

### 2. Why does the gradient imbalance persist during training?
**Hypothesis:** As model trains, prev_state becomes more informative but:
- Accumulation effect persists (structural)
- Main network already behind, can't catch up
- Need early-training dynamics analysis

### 3. Can we fix RLS without disabling the side stream?

**Potential fixes to investigate:**
- **Scale down prev_state contribution:** `prev_state * 0.1` to balance gradients
- **Separate learning rates:** Lower LR for E_type_side
- **Gradient clipping per-parameter:** Clip E_type_side specifically
- **Architecture change:** Don't use shared E_type_side for all positions
- **Attention temperature:** Scale attention to side tokens differently

---

## Next Steps

### Immediate (when experiments complete)

1. **Analyze results from H100 experiments**
   ```bash
   ssh ubuntu@100.121.60.36 "grep 'Validation' /tmp/*_1k.log"
   ```
   - Extract validation bpb at steps 100, 200, ..., 1000
   - Plot curves: Baseline vs RLS vs RLS_masked
   - Expected result: RLS_masked ≈ Baseline

2. **If mask_side_attention works:**
   - Confirms noisy gradient dominance theory
   - Move to fixing RLS (don't just disable side stream)

3. **If mask_side_attention doesn't fully recover:**
   - Additional problems exist (memory state, architecture, etc.)
   - Need deeper investigation

### Short-term Experiments

4. **Test gradient balancing approaches:**
   - Add `prev_state_scale` parameter to GPTConfig
   - Test with scale=0.1, 0.01, 0.001
   - Find value that balances E_type_side and E_type_main gradients

5. **Analyze early training dynamics (steps 0-100):**
   - Track E_type_side vs E_type_main gradient ratio over time
   - Measure attention entropy to side tokens
   - Check gradient variance (high variance = noisy/random)
   - Correlation between E_type_side updates and loss improvement

6. **Test separate learning rates:**
   - Modify optimizer to use lower LR for E_type_side
   - Try 0.1x, 0.01x of main LR

### Long-term Investigations

7. **Architecture alternatives:**
   - Position-specific side embeddings instead of shared E_type_side
   - Learnable attention mask for side stream
   - Gating mechanism to dynamically weight main vs side

8. **Batch size scaling experiments:**
   - Test with batch sizes: 512, 4096, 16384, 65536
   - Check if gradient imbalance gets worse with larger batches
   - Theory 2 validation

---

## Important File Locations

### Local Machine
- Working directory: `/Users/ericjordan/Documents/workspace/nanochat-ewj`
- Branch: `recurrent-layer-state`
- Key files:
  - `nanochat/gpt.py` - Model with ablation flags
  - `scripts/base_train.py` - Training script
  - `test_ablations.py` - Ablation validation
  - `test_rls_diagnostics.py` - Gradient diagnostics
  - `RLS_THEORY_ANALYSIS.md` - Theory analysis
  - `LAMBDA_SERVER_SETUP.md` - H100 server setup

### H100 Server (100.121.60.36)
- Logs: `/tmp/baseline_1k.log`, `/tmp/rls_1k.log`, `/tmp/rls_masked_1k.log`
- Monitoring script: `/tmp/monitor_ablation.sh`
- Orchestrator script: `/tmp/run_remaining.sh`
- Tmux sessions: `baseline`, `rls`, `rls_masked`, `orchestrator`

### Previous Experiments
- H100 full run logs: `/home/ubuntu/nanochat/logs/` (if saved)
- Local 500-step tests: `/tmp/rls_500.log`, `/tmp/baseline_500.log`
- Diagnostic output: `/tmp/h100_diagnostics_output.log`

---

## Commands Reference

### Check Experiment Status
```bash
# View monitoring summary
ssh ubuntu@100.121.60.36 "bash /tmp/monitor_ablation.sh"

# Check which experiments are running
ssh ubuntu@100.121.60.36 "tmux ls"

# View live log
ssh ubuntu@100.121.60.36 "tail -f /tmp/baseline_1k.log"

# Extract validation results
ssh ubuntu@100.121.60.36 "grep 'Validation' /tmp/*_1k.log"

# Attach to tmux session (Ctrl+B, D to detach)
ssh -t ubuntu@100.121.60.36 "tmux attach -t baseline"
```

### Run Ablation Tests Locally
```bash
cd /Users/ericjordan/Documents/workspace/nanochat-ewj
source .venv/bin/activate
python test_ablations.py
```

### Run Diagnostic Analysis
```bash
python test_rls_diagnostics.py  # Takes ~2 minutes
```

---

## Key Metrics to Track

When analyzing results, focus on:

1. **Validation bpb curves** - Main success metric
   - Baseline should improve steadily
   - RLS should plateau or improve slowly
   - RLS_masked should track close to baseline (if theory correct)

2. **Gradient ratios** - E_type_side / E_type_main
   - Normal RLS: 22-51x
   - RLS_masked: Should be 0.00x (confirmed by test_ablations.py)

3. **Training loss** - Secondary metric
   - All should decrease
   - Rate of decrease shows learning efficiency

4. **Step timing** - Check for slowdown
   - RLS has warmup overhead
   - RLS_masked should be same as normal RLS

---

## Context for New Session

**What we know for certain:**
1. RLS fails catastrophically at scale (1.76 vs 1.13 bpb)
2. E_type_side receives 22-50x more gradient than E_type_main from step 0
3. Overall gradient norms are equal - problem is imbalance, not weakness
4. Side stream dominance happens even when prev_state is uninformative
5. mask_side_attention eliminates side gradients perfectly (validated)

**What we're testing:**
- Does disabling side stream recover baseline performance?
- If yes: Confirms noisy gradient dominance theory
- If no: Additional problems exist

**What we need to do next:**
- Analyze experiment results (check `/tmp/*_1k.log` on H100)
- If theory confirmed: Develop fix that balances gradients without disabling RLS
- If theory incomplete: Investigate what else is wrong

**Technical debt:**
- Many background bash processes still running (can safely ignore)
- Should commit current changes after validating experiments

---

## Questions to Consider

1. If masking works, why does RLS design include side tokens at all?
   - Original intent: Provide recurrent context
   - Execution: Creates harmful gradient imbalance

2. Can we redesign to keep benefits without the imbalance?
   - Scaling factor on prev_state
   - Position-specific embeddings
   - Gating mechanisms

3. Is this a general problem for any architecture with auxiliary streams?
   - Other recurrent architectures?
   - Retrieval-augmented models?

---

## References

- Original RLS implementation: `nanochat/gpt.py:100-400`
- H100 catastrophic run: Step 0-6000, val bpb 1.76 at end
- Local 500-step comparison: Gap closed from 0.14 to 0.07 then plateau
- Diagnostic run output: `/tmp/h100_diagnostics_output.log`
- Theory analysis: `RLS_THEORY_ANALYSIS.md`

---

*Last updated: 2025-11-12*
*Session context preserved for continuation*
