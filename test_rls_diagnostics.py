#!/usr/bin/env python3
"""
RLS Diagnostic Script - Test Three Theories

Tests three theories for RLS failure:
1. Side Token Attention Dominance
2. Batch Size Scaling Problem
3. Gradient Flow Architecture Issue

Logs:
- Attention patterns (main vs side)
- Layer-wise gradient norms
- Type embedding gradients
- Embedding vs layer gradient comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    num_recurrence_warmup=1,
)

def check_gradient_flow(model, x, y):
    """Check gradients for all layers and RLS components"""
    model.train()
    model.zero_grad()

    # Forward + backward
    loss = model.forward_with_recurrence(x, y)
    loss.backward()

    # Collect layer-wise gradient norms
    layer_grads = {}
    for i, block in enumerate(model.transformer.h):
        attn_grads = [p.grad.norm().item() for p in block.attn.parameters() if p.grad is not None]
        mlp_grads = [p.grad.norm().item() for p in block.mlp.parameters() if p.grad is not None]
        layer_grads[f'layer{i}_attn'] = sum(attn_grads) / max(len(attn_grads), 1)
        layer_grads[f'layer{i}_mlp'] = sum(mlp_grads) / max(len(mlp_grads), 1)

    # RLS component gradients
    rls_grads = {
        'E_type_main': model.E_type_main.grad.norm().item() if model.E_type_main.grad is not None else 0.0,
        'E_type_side': model.E_type_side.grad.norm().item() if model.E_type_side.grad is not None else 0.0,
    }

    # Embedding gradients
    wte_grad = model.transformer.wte.weight.grad.norm().item() if model.transformer.wte.weight.grad is not None else 0.0

    # Overall gradient norm
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()

    return {
        'loss': loss.item(),
        'total_grad_norm': total_grad_norm,
        'layer_grads': layer_grads,
        'rls_grads': rls_grads,
        'wte_grad': wte_grad,
    }


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


def run_training_steps(model, data_loader, optimizer, steps=10):
    """Run several training steps and collect metrics"""
    metrics = []

    for step in range(steps):
        x, y = next(data_loader)
        stats = check_gradient_flow(model, x, y)
        optimizer.step()
        metrics.append(stats)

    return metrics


def main():
    print("=" * 80)
    print("RLS Gradient Flow Diagnostics - Testing Three Theories")
    print("=" * 80)
    print()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = get_tokenizer()
    print(f"Loaded tokenizer (vocab_size={tokenizer.get_vocab_size()})")

    # Create RLS model
    print("Creating RLS model...")
    model = GPT(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create baseline for comparison
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
    print(f"Baseline parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    print()

    # Setup data loader
    print("Loading data...")
    data_loader = tokenizing_distributed_data_loader(
        B=1,  # Small batch for fast testing
        T=512,
        split="train",
        tokenizer_threads=1,
        device=device,
    )

    # Get a single batch for testing
    x, y = next(data_loader)
    print(f"Batch shape: {x.shape}")
    print()

    # =================================================================
    # Test: Gradient Flow Comparison (Theory 3)
    # =================================================================
    print("=" * 80)
    print("TEST: Gradient Flow Comparison (RLS vs Baseline)")
    print("=" * 80)
    print()

    print("RLS Model:")
    rls_stats = check_gradient_flow(model, x, y)
    print(f"  Loss: {rls_stats['loss']:.6f}")
    print(f"  Total grad norm: {rls_stats['total_grad_norm']:.6f}")
    print(f"  wte grad: {rls_stats['wte_grad']:.6f}")
    print(f"  E_type_main grad: {rls_stats['rls_grads']['E_type_main']:.6f}")
    print(f"  E_type_side grad: {rls_stats['rls_grads']['E_type_side']:.6f}")
    print(f"  E_type ratio (side/main): {rls_stats['rls_grads']['E_type_side'] / max(rls_stats['rls_grads']['E_type_main'], 1e-10):.2f}x")
    print()
    print("  Layer-wise gradients (attention):")
    for i in range(12):
        print(f"    Layer {i:2d}: {rls_stats['layer_grads'][f'layer{i}_attn']:.6f}")
    print()

    print("Baseline Model:")
    baseline_model.zero_grad()
    baseline_loss = baseline_model(x, y)
    baseline_loss.backward()
    baseline_grad_norm = torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), float('inf')).item()
    baseline_wte_grad = baseline_model.transformer.wte.weight.grad.norm().item()
    baseline_layer0_grads = [p.grad.norm().item() for p in baseline_model.transformer.h[0].attn.parameters() if p.grad is not None]
    baseline_layer0_attn = sum(baseline_layer0_grads) / max(len(baseline_layer0_grads), 1)

    print(f"  Loss: {baseline_loss.item():.6f}")
    print(f"  Total grad norm: {baseline_grad_norm:.6f}")
    print(f"  wte grad: {baseline_wte_grad:.6f}")
    print(f"  Layer 0 attn grad: {baseline_layer0_attn:.6f}")
    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()

    print("Theory 1: Side Token Attention Dominance")
    E_type_ratio = rls_stats['rls_grads']['E_type_side'] / max(rls_stats['rls_grads']['E_type_main'], 1e-10)
    if E_type_ratio > 10:
        print(f"  ⚠️  E_type_side grad is {E_type_ratio:.1f}x larger than E_type_main")
        print("  This suggests side tokens might be dominating attention")
    else:
        print(f"  ✓ E_type gradients are relatively balanced ({E_type_ratio:.1f}x)")
    print()

    print("Theory 2: Batch Size Scaling")
    print("  (Requires running with different batch sizes: 512, 8192, 65536)")
    print(f"  Current batch size: 512")
    print()

    print("Theory 3: Gradient Flow Architecture Issue")
    grad_ratio = rls_stats['total_grad_norm'] / baseline_grad_norm
    layer0_ratio = rls_stats['layer_grads']['layer0_attn'] / baseline_layer0_attn
    wte_ratio = rls_stats['wte_grad'] / baseline_wte_grad

    print(f"  Overall gradient strength (RLS/Baseline): {grad_ratio:.2f}x")
    print(f"  Layer 0 gradient strength (RLS/Baseline): {layer0_ratio:.2f}x")
    print(f"  Embedding gradient strength (RLS/Baseline): {wte_ratio:.2f}x")

    if grad_ratio < 0.7:
        print(f"  ⚠️  RLS has {(1-grad_ratio)*100:.0f}% weaker gradients overall")
        print("  This suggests fundamental gradient flow issue")
    else:
        print("  ✓ Gradient strengths are comparable")

    if layer0_ratio < 0.5:
        print(f"  ⚠️  Layer 0 has particularly weak gradients in RLS")
        print("  This confirms dual-stream attention dampens gradients")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("Based on this single-step analysis:")
    if grad_ratio < 0.7:
        print("  → Theory 3 (Gradient Flow) is SUPPORTED")
        print(f"    RLS shows systematically weaker gradients ({(1-grad_ratio)*100:.0f}% weaker)")
    if E_type_ratio > 10:
        print("  → Theory 1 (Side Dominance) is SUPPORTED")
        print(f"    E_type_side receives {E_type_ratio:.1f}x more gradient than E_type_main")
    if grad_ratio >= 0.7 and E_type_ratio <= 10:
        print("  → No clear gradient flow issue detected")
        print("    Theory 2 (Batch Size) may be more relevant")
    print()


if __name__ == "__main__":
    main()
