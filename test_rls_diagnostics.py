#!/usr/bin/env python3
"""
RLS Diagnostic Script - Investigate training failure

Tests different hypotheses:
1. Gradient flow to RLS components
2. Warmup iteration count (1 vs 3 vs 5)
3. Prev_state magnitude/distribution
4. Learning dynamics comparison
"""
import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader
import os

# Simple config for fast iteration
config = GPTConfig(
    vocab_size=65536,
    n_layer=12,
    n_embd=768,
    n_head=6,
    n_kv_head=6,
    sequence_len=512,
    recurrent_layer_state=True,
    num_recurrence_warmup=1,  # Will vary this
)

def check_gradient_flow(model, x, y, warmup_iters=1):
    """Check if RLS components receive gradients"""
    model.config.num_recurrence_warmup = warmup_iters
    model.train()

    # Zero gradients
    model.zero_grad()

    # Forward + backward
    loss = model.forward_with_recurrence(x, y)
    loss.backward()

    # Check RLS component gradients
    results = {
        'E_type_main': model.E_type_main.grad,
        'E_type_side': model.E_type_side.grad,
        'side_mlp.0.weight': model.side_mlp[0].weight.grad,
        'side_mlp.2.weight': model.side_mlp[2].weight.grad,
    }

    stats = {}
    for name, grad in results.items():
        if grad is None:
            stats[name] = {'exists': False}
        else:
            stats[name] = {
                'exists': True,
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
            }

    return loss.item(), stats


def check_prev_state_stats(model, x, warmup_iters=1):
    """Check prev_state distribution during warmup"""
    model.config.num_recurrence_warmup = warmup_iters
    model.eval()

    B, T = x.size()
    device = x.device
    model_dtype = next(model.parameters()).dtype

    prev_state = torch.zeros(B, T, model.config.n_embd, dtype=model_dtype, device=device)

    states_over_warmup = []

    with torch.no_grad():
        for i in range(warmup_iters):
            _, warmup_state = model.forward(x, targets=None, prev_state=prev_state, return_state=True)
            states_over_warmup.append({
                'iteration': i,
                'mean': warmup_state.mean().item(),
                'std': warmup_state.std().item(),
                'max': warmup_state.abs().max().item(),
            })
            # Shift for next iteration
            prev_state = torch.cat([
                torch.zeros(B, 1, model.config.n_embd, dtype=model_dtype, device=device),
                warmup_state[:, :-1, :]
            ], dim=1)

    return prev_state, states_over_warmup


def training_step_comparison(model, x, y, warmup_iters=1):
    """Compare a training step with different warmup settings"""
    model.config.num_recurrence_warmup = warmup_iters
    model.train()
    model.zero_grad()

    # Measure time and loss
    import time
    start = time.time()
    loss = model.forward_with_recurrence(x, y)
    loss.backward()
    elapsed = time.time() - start

    # Gather gradient stats for all parameters
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()

    return {
        'warmup_iters': warmup_iters,
        'loss': loss.item(),
        'time': elapsed,
        'rls_grad_norms': {
            'E_type_main': grad_norms.get('E_type_main', 0),
            'E_type_side': grad_norms.get('E_type_side', 0),
            'side_mlp.0.weight': grad_norms.get('side_mlp.0.weight', 0),
            'side_mlp.2.weight': grad_norms.get('side_mlp.2.weight', 0),
        }
    }


def main():
    print("=" * 60)
    print("RLS Diagnostic Tests")
    print("=" * 60)
    print()

    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load tokenizer
    base_dir = os.path.expanduser("~/.cache/nanochat")
    tokenizer = get_tokenizer()
    print(f"Loaded tokenizer (vocab_size={tokenizer.get_vocab_size()})")

    # Create model
    print("Creating model...")
    model = GPT(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Get a batch of data
    print("Loading data batch...")
    data_loader = tokenizing_distributed_data_loader(
        B=2,  # Small batch for speed
        T=512,
        split="train",
        tokenizer_threads=1,
        device=device
    )
    x, y = next(data_loader)
    print(f"Batch shape: {x.shape}")
    print()

    # Test 1: Gradient flow check
    print("=" * 60)
    print("TEST 1: Gradient Flow to RLS Components")
    print("=" * 60)
    for warmup in [1, 3, 5]:
        print(f"\nWarmup iterations: {warmup}")
        loss, grad_stats = check_gradient_flow(model, x, y, warmup_iters=warmup)
        print(f"Loss: {loss:.4f}")
        for name, stats in grad_stats.items():
            if stats['exists']:
                print(f"  {name:20s}: mean={stats['mean']:+.6f}, std={stats['std']:.6f}, max={stats['max']:.6f}")
            else:
                print(f"  {name:20s}: NO GRADIENT")

    # Test 2: Prev_state statistics
    print("\n" + "=" * 60)
    print("TEST 2: Prev_state Evolution During Warmup")
    print("=" * 60)
    for warmup in [1, 3, 5]:
        print(f"\nWarmup iterations: {warmup}")
        final_state, state_history = check_prev_state_stats(model, x, warmup_iters=warmup)
        for stats in state_history:
            i = stats['iteration']
            print(f"  Iteration {i}: mean={stats['mean']:+.6f}, std={stats['std']:.6f}, max={stats['max']:.6f}")
        print(f"  Final prev_state: mean={final_state.mean().item():+.6f}, std={final_state.std().item():.6f}")

    # Test 3: Training step comparison
    print("\n" + "=" * 60)
    print("TEST 3: Training Step Comparison (5 steps each)")
    print("=" * 60)

    for warmup in [1, 3, 5]:
        print(f"\nWarmup iterations: {warmup}")
        model = GPT(config).to(device)  # Fresh model
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        losses = []
        for step in range(5):
            stats = training_step_comparison(model, x, y, warmup_iters=warmup)
            optimizer.step()
            losses.append(stats['loss'])

            if step == 0:
                print(f"  Step {step}: loss={stats['loss']:.4f}, time={stats['time']:.3f}s")
                print(f"    RLS grad norms: E_type_main={stats['rls_grad_norms']['E_type_main']:.6f}, "
                      f"side_mlp.0={stats['rls_grad_norms']['side_mlp.0.weight']:.6f}")

        print(f"  Loss progression: {losses[0]:.4f} → {losses[-1]:.4f} (delta: {losses[-1]-losses[0]:+.4f})")

    # Test 4: Baseline comparison (no RLS)
    print("\n" + "=" * 60)
    print("TEST 4: Baseline (no RLS) Comparison")
    print("=" * 60)
    baseline_config = GPTConfig(
        vocab_size=65536,
        n_layer=12,
        n_embd=768,
        n_head=6,
        n_kv_head=6,
        sequence_len=512,
        recurrent_layer_state=False,
    )
    baseline_model = GPT(baseline_config).to(device)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=3e-4)

    baseline_losses = []
    for step in range(5):
        baseline_model.zero_grad()
        loss = baseline_model(x, y)
        loss.backward()
        baseline_optimizer.step()
        baseline_losses.append(loss.item())

    print(f"Baseline loss progression: {baseline_losses[0]:.4f} → {baseline_losses[-1]:.4f} (delta: {baseline_losses[-1]-baseline_losses[0]:+.4f})")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Check the results above to diagnose:")
    print("1. Are RLS gradients flowing? (magnitudes should be non-zero)")
    print("2. Does warmup iteration count affect prev_state quality?")
    print("3. Does RLS learn as fast as baseline in early steps?")
    print("4. Are gradient norms reasonable compared to other parameters?")


if __name__ == "__main__":
    main()
