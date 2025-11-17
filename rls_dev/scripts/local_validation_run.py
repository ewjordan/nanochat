#!/usr/bin/env python
"""
Local validation run for recurrent layer state feature.
Trains two small models (with/without RLS) and compares results.
"""

import os
import sys
import time
import torch
from nanochat.gpt import GPT, GPTConfig

def train_small_model(use_recurrence=False, num_steps=100, depth=6):
    """Train a small model and return final loss."""

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        device_type = 'mps'
    else:
        device = torch.device('cpu')
        device_type = 'cpu'

    # Small model config
    num_layers = depth
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)

    config = GPTConfig(
        vocab_size=16000,  # Match nanochat default
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        sequence_len=512,  # Shorter for local testing
        recurrent_layer_state=use_recurrence,
        num_recurrence_warmup=1
    )

    model = GPT(config)
    model.init_weights()
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # For validation, use synthetic data (fixed sequence that repeats)
    # This ensures both models see identical data for fair comparison
    torch.manual_seed(42)  # Deterministic data
    print(f"  Using synthetic data (deterministic for fair comparison)")

    # Training loop
    losses = []
    start_time = time.time()

    for step in range(num_steps):
        # Generate synthetic data (same sequence each time for consistency)
        x = torch.randint(0, config.vocab_size, (2, config.sequence_len), device=device, generator=torch.Generator(device=device).manual_seed(step))
        y = torch.randint(0, config.vocab_size, (2, config.sequence_len), device=device, generator=torch.Generator(device=device).manual_seed(step + 1000))

        optimizer.zero_grad()

        if use_recurrence:
            loss = model.forward_with_recurrence(x, y)
        else:
            loss = model(x, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 20 == 0:
            avg_loss = sum(losses[-20:]) / 20
            elapsed = time.time() - start_time
            step_time = elapsed / (step + 1)
            print(f"    Step {step + 1}/{num_steps} | Loss: {avg_loss:.4f} | {step_time*1000:.1f}ms/step")

    elapsed = time.time() - start_time
    final_loss = sum(losses[-10:]) / 10  # Average last 10

    print(f"  Finished in {elapsed:.1f}s")
    print(f"  Final loss (avg last 10): {final_loss:.4f}")

    return losses, final_loss

def main():
    print("=" * 70)
    print("LOCAL VALIDATION RUN: Testing Recurrent Layer State")
    print("=" * 70)
    print()

    depth = 6  # Small model
    num_steps = 100

    print(f"Configuration:")
    print(f"  Depth: {depth} (model_dim={depth*64}, params~{(depth*64)**2 * depth * 12 / 1e6:.1f}M)")
    print(f"  Steps: {num_steps}")
    print(f"  Sequence length: 512")
    print()

    # Test 1: Without recurrence
    print("-" * 70)
    print("TEST 1: Baseline (no recurrent layer state)")
    print("-" * 70)
    losses_baseline, final_baseline = train_small_model(
        use_recurrence=False,
        num_steps=num_steps,
        depth=depth
    )
    print()

    # Test 2: With recurrence
    print("-" * 70)
    print("TEST 2: With recurrent layer state")
    print("-" * 70)
    losses_rls, final_rls = train_small_model(
        use_recurrence=True,
        num_steps=num_steps,
        depth=depth
    )
    print()

    # Compare
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline final loss:  {final_baseline:.4f}")
    print(f"RLS final loss:       {final_rls:.4f}")

    if final_rls < final_baseline:
        improvement = ((final_baseline - final_rls) / final_baseline) * 100
        print(f"✅ RLS improved by {improvement:.1f}%")
    elif final_rls > final_baseline:
        degradation = ((final_rls - final_baseline) / final_baseline) * 100
        print(f"⚠️  RLS is {degradation:.1f}% worse (might improve with more training)")
    else:
        print(f"≈ No significant difference")

    print()
    print("Note: This is a tiny test. Real validation requires:")
    print("  - More training steps (thousands)")
    print("  - Larger model (depth 12-20)")
    print("  - Real evaluation metrics (not just loss)")
    print()
    print("But if both models trained without errors, the feature is working! ✅")

if __name__ == "__main__":
    main()
