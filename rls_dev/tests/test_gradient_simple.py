"""Simple test to understand gradient flow through state_gate."""

import torch
import torch.nn as nn

def test_simple_gradient():
    """Test gradient flow in a minimal example."""

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Testing on device: {device}\n")

    # Create a simple linear layer like state_gate
    n_embd = 4
    state_gate = nn.Linear(2 * n_embd, n_embd, bias=False).to(device)

    # Initialize like we do in the real code
    with torch.no_grad():
        state_gate.weight[:, :n_embd] = torch.eye(n_embd)
        state_gate.weight[:, n_embd:] = 0.0

    print("Initial weights:")
    print(state_gate.weight)
    print()

    # Create inputs
    token_emb = torch.randn(2, 3, n_embd, device=device, requires_grad=True)
    prev_state_nograd = torch.randn(2, 3, n_embd, device=device)  # No gradient
    prev_state_grad = torch.randn(2, 3, n_embd, device=device, requires_grad=True)

    print(f"token_emb requires_grad: {token_emb.requires_grad}")
    print(f"prev_state_nograd requires_grad: {prev_state_nograd.requires_grad}")
    print(f"prev_state_grad requires_grad: {prev_state_grad.requires_grad}")
    print()

    # Test 1: With prev_state that has no gradient (like in warmup)
    print("=" * 60)
    print("Test 1: prev_state with NO gradient (like in forward_with_recurrence)")
    print("=" * 60)

    gate_input_nograd = torch.cat([token_emb, prev_state_nograd], dim=-1)
    output_nograd = state_gate(gate_input_nograd)

    # Compute a simple loss
    loss_nograd = output_nograd.sum()
    loss_nograd.backward()

    print(f"Loss: {loss_nograd.item():.4f}")
    print(f"state_gate gradient norm: {state_gate.weight.grad.norm().item():.6f}")
    print(f"state_gate gradient max: {state_gate.weight.grad.abs().max().item():.6f}")
    print(f"Gradient sample (first row): {state_gate.weight.grad[0, :]}")
    print()

    # Test 2: With prev_state that has gradient
    print("=" * 60)
    print("Test 2: prev_state WITH gradient (for comparison)")
    print("=" * 60)

    state_gate.zero_grad()
    token_emb.grad = None

    gate_input_grad = torch.cat([token_emb, prev_state_grad], dim=-1)
    output_grad = state_gate(gate_input_grad)

    loss_grad = output_grad.sum()
    loss_grad.backward()

    print(f"Loss: {loss_grad.item():.4f}")
    print(f"state_gate gradient norm: {state_gate.weight.grad.norm().item():.6f}")
    print(f"state_gate gradient max: {state_gate.weight.grad.abs().max().item():.6f}")
    print(f"Gradient sample (first row): {state_gate.weight.grad[0, :]}")
    print()

    # Test 3: What if we use a non-identity initialization?
    print("=" * 60)
    print("Test 3: With random initialization (not identity)")
    print("=" * 60)

    state_gate2 = nn.Linear(2 * n_embd, n_embd, bias=False).to(device)
    # Use default random initialization

    token_emb2 = torch.randn(2, 3, n_embd, device=device, requires_grad=True)
    prev_state2 = torch.randn(2, 3, n_embd, device=device)  # No gradient

    gate_input2 = torch.cat([token_emb2, prev_state2], dim=-1)
    output2 = state_gate2(gate_input2)

    loss2 = output2.sum()
    loss2.backward()

    print(f"Loss: {loss2.item():.4f}")
    print(f"state_gate gradient norm: {state_gate2.weight.grad.norm().item():.6f}")
    print(f"state_gate gradient max: {state_gate2.weight.grad.abs().max().item():.6f}")
    print()

if __name__ == "__main__":
    test_simple_gradient()
