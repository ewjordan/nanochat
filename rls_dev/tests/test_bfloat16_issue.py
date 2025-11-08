"""Test if bfloat16 causes gradient issues on MPS."""

import torch
import torch.nn as nn

def test_bfloat16_gradients():
    """Test gradient flow with bfloat16 vs float32."""

    # Detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Testing on device: {device}\n")

    n_embd = 64
    state_gate = nn.Linear(2 * n_embd, n_embd, bias=False).to(device)

    # Initialize like our real code
    with torch.no_grad():
        state_gate.weight[:, :n_embd] = torch.eye(n_embd)
        state_gate.weight[:, n_embd:] = 0.0

    # Test with bfloat16 prev_state
    print("=" * 60)
    print("Test with bfloat16 prev_state (like current code)")
    print("=" * 60)

    token_emb = torch.randn(2, 10, n_embd, device=device, requires_grad=True)
    prev_state_bf16 = torch.zeros(2, 10, n_embd, dtype=torch.bfloat16, device=device)

    gate_input = torch.cat([token_emb, prev_state_bf16], dim=-1)
    output = state_gate(gate_input)
    loss = output.sum()
    loss.backward()

    print(f"token_emb dtype: {token_emb.dtype}")
    print(f"prev_state dtype: {prev_state_bf16.dtype}")
    print(f"gate_input dtype: {gate_input.dtype}")
    print(f"output dtype: {output.dtype}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Gradient norm: {state_gate.weight.grad.norm().item():.6f}")
    print(f"Gradient max: {state_gate.weight.grad.abs().max().item():.6f}")
    print()

    # Test with float32 prev_state
    print("=" * 60)
    print("Test with float32 prev_state (proposed fix)")
    print("=" * 60)

    state_gate.zero_grad()
    token_emb.grad = None

    token_emb2 = torch.randn(2, 10, n_embd, device=device, requires_grad=True)
    prev_state_f32 = torch.zeros(2, 10, n_embd, dtype=torch.float32, device=device)

    gate_input2 = torch.cat([token_emb2, prev_state_f32], dim=-1)
    output2 = state_gate(gate_input2)
    loss2 = output2.sum()
    loss2.backward()

    print(f"token_emb dtype: {token_emb2.dtype}")
    print(f"prev_state dtype: {prev_state_f32.dtype}")
    print(f"gate_input dtype: {gate_input2.dtype}")
    print(f"output dtype: {output2.dtype}")
    print(f"Loss: {loss2.item():.4f}")
    print(f"Gradient norm: {state_gate.weight.grad.norm().item():.6f}")
    print(f"Gradient max: {state_gate.weight.grad.abs().max().item():.6f}")
    print()

    # Test with non-zero bfloat16 prev_state
    print("=" * 60)
    print("Test with NON-ZERO bfloat16 prev_state")
    print("=" * 60)

    state_gate.zero_grad()

    token_emb3 = torch.randn(2, 10, n_embd, device=device, requires_grad=True)
    prev_state_nonzero = torch.randn(2, 10, n_embd, device=device).to(torch.bfloat16)

    gate_input3 = torch.cat([token_emb3, prev_state_nonzero], dim=-1)
    output3 = state_gate(gate_input3)
    loss3 = output3.sum()
    loss3.backward()

    print(f"prev_state mean: {prev_state_nonzero.float().abs().mean().item():.6f}")
    print(f"Loss: {loss3.item():.4f}")
    print(f"Gradient norm: {state_gate.weight.grad.norm().item():.6f}")
    print(f"Gradient max: {state_gate.weight.grad.abs().max().item():.6f}")

if __name__ == "__main__":
    test_bfloat16_gradients()
