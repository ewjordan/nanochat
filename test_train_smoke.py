"""
Smoke test for RLS training - verify model can train for a few steps without errors.
"""
import torch
import sys
sys.path.insert(0, '.')

from nanochat.gpt import GPT, GPTConfig

def test_training_smoke():
    """Test that RLS model can train for a few steps."""
    print("Training Smoke Test for RLS Side Tokens...")

    # Create small model with RLS
    config = GPTConfig(
        vocab_size=1024,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=256,
        sequence_len=128,
        recurrent_layer_state=True,
        num_recurrence_warmup=1
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GPT(config).to(device)
    model.init_weights()
    model.train()

    # Create optimizer (simple AdamW for test)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Prepare dummy data
    B, T = 4, 64

    print("\nTraining baseline (no RLS) for 10 steps...")
    baseline_losses = []
    for step in range(10):
        idx = torch.randint(0, config.vocab_size, (B, T), device=device)
        targets = torch.randint(0, config.vocab_size, (B, T), device=device)

        optimizer.zero_grad()
        loss = model(idx, targets=targets, prev_state=None)
        loss.backward()
        optimizer.step()

        baseline_losses.append(loss.item())
        if step % 5 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    print(f"  Final baseline loss: {baseline_losses[-1]:.4f}")

    print("\nTraining with RLS (warmup + gradient) for 10 steps...")
    rls_losses = []
    for step in range(10):
        idx = torch.randint(0, config.vocab_size, (B, T), device=device)
        targets = torch.randint(0, config.vocab_size, (B, T), device=device)

        # Use forward_with_recurrence for RLS
        optimizer.zero_grad()
        loss = model.forward_with_recurrence(idx, targets=targets)
        loss.backward()

        # Check for None gradients
        none_grads = []
        for name, param in model.named_parameters():
            if param.grad is None:
                none_grads.append(name)

        if none_grads:
            print(f"  WARNING: Step {step} has None gradients: {none_grads[:3]}")

        optimizer.step()

        rls_losses.append(loss.item())
        if step % 5 == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}, none_grads = {len(none_grads)}")

    print(f"  Final RLS loss: {rls_losses[-1]:.4f}")

    # Check gradients flow to new parameters
    print("\nChecking gradients on RLS-specific parameters...")
    idx = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)

    optimizer.zero_grad()
    loss = model.forward_with_recurrence(idx, targets=targets)
    loss.backward()

    rls_param_grads = {
        'E_type_main': model.E_type_main.grad is not None,
        'E_type_side': model.E_type_side.grad is not None,
        'side_mlp.0': model.side_mlp[0].weight.grad is not None,
        'side_mlp.2': model.side_mlp[2].weight.grad is not None,
    }

    for name, has_grad in rls_param_grads.items():
        status = "✓" if has_grad else "✗"
        print(f"  {status} {name}: grad = {has_grad}")

    all_have_grads = all(rls_param_grads.values())

    print("\n" + "="*60)
    if all_have_grads:
        print("✓ Smoke test passed!")
        print("  - Baseline training works")
        print("  - RLS training works")
        print("  - Gradients flow to all RLS parameters")
    else:
        print("✗ Smoke test FAILED - some RLS parameters don't have gradients")
        return False
    print("="*60)

    return True

if __name__ == "__main__":
    success = test_training_smoke()
    sys.exit(0 if success else 1)
