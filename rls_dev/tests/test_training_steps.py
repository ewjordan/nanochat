"""Test that state_gate learns over multiple training steps."""

import torch
from nanochat.gpt import GPT, GPTConfig

def test_multi_step_training():
    """Run multiple training steps and verify state_gate starts learning."""

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Testing on device: {device}\n")

    config = GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        recurrent_layer_state=True,
        num_recurrence_warmup=1
    )

    model = GPT(config)
    model.init_weights()
    model = model.to(device)

    # Simple optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("Running 5 training steps...\n")

    for step in range(5):
        # Create random training data
        B, T = 4, 20
        idx = torch.randint(0, config.vocab_size, (B, T), device=device)
        targets = torch.randint(0, config.vocab_size, (B, T), device=device)

        # Forward pass
        optimizer.zero_grad()
        loss = model.forward_with_recurrence(idx, targets)
        loss.backward()
        optimizer.step()

        # Check gradients
        state_gate_grad_norm = model.state_gate.weight.grad.norm().item() if model.state_gate.weight.grad is not None else 0.0
        wte_grad_norm = model.transformer.wte.weight.grad.norm().item() if model.transformer.wte.weight.grad is not None else 0.0
        mlp_grad_norm = model.transformer.h[0].mlp.c_fc.weight.grad.norm().item() if model.transformer.h[0].mlp.c_fc.weight.grad is not None else 0.0

        print(f"Step {step + 1}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  state_gate grad norm: {state_gate_grad_norm:.6f}")
        print(f"  wte grad norm: {wte_grad_norm:.6f}")
        print(f"  MLP grad norm: {mlp_grad_norm:.6f}")

    print("\n" + "=" * 60)
    if state_gate_grad_norm > 1e-6:
        print("✅ state_gate is learning (gradients are non-zero)")
    else:
        print("⚠️  state_gate gradients still near zero after 5 steps")

if __name__ == "__main__":
    test_multi_step_training()
