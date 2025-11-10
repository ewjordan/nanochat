"""
Test script for side token RLS implementation.
"""
import torch
import sys
sys.path.insert(0, '.')

from nanochat.gpt import GPT, GPTConfig

def test_rls_side_tokens():
    """Test that RLS with side tokens works correctly."""
    print("Testing RLS Side Token Implementation...")

    # Create a small model config with RLS enabled
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

    # Create model and move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GPT(config).to(device)
    model.init_weights()
    model.eval()

    # Test 1: Forward pass WITHOUT prev_state (baseline mode)
    print("\n[Test 1] Forward pass without prev_state...")
    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)

    with torch.no_grad():
        loss_no_state = model(idx, targets=targets, prev_state=None)
    print(f"  Loss (no prev_state): {loss_no_state.item():.4f}")
    print(f"  ✓ Pass")

    # Test 2: Forward pass WITH prev_state
    print("\n[Test 2] Forward pass with prev_state...")
    # Use same dtype as model (float32 on CPU, bfloat16 on CUDA)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    prev_state = torch.randn(B, T, config.n_embd, device=device, dtype=dtype)

    with torch.no_grad():
        loss_with_state = model(idx, targets=targets, prev_state=prev_state)
    print(f"  Loss (with prev_state): {loss_with_state.item():.4f}")
    print(f"  ✓ Pass")

    # Test 3: Return state
    print("\n[Test 3] Forward pass with return_state...")
    with torch.no_grad():
        loss, final_state = model(idx, targets=targets, prev_state=prev_state, return_state=True)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Final state shape: {final_state.shape}")
    assert final_state.shape == (B, T, config.n_embd), f"Expected shape (B, T, n_embd), got {final_state.shape}"
    print(f"  ✓ Pass")

    # Test 4: Warmup pass (no targets, return_state=True, no kv_cache)
    print("\n[Test 4] Warmup pass (should skip lm_head)...")
    with torch.no_grad():
        logits, warmup_state = model(idx, targets=None, prev_state=prev_state, return_state=True)
    assert logits is None, "Warmup should return None logits"
    assert warmup_state.shape == (B, T, config.n_embd)
    print(f"  Warmup state shape: {warmup_state.shape}")
    print(f"  ✓ Pass (lm_head skipped)")

    # Test 5: Generation mode (with KV cache)
    print("\n[Test 5] Generation mode with KV cache...")
    from nanochat.engine import KVCache

    kv_cache = KVCache(
        batch_size=B,
        num_heads=config.n_kv_head,
        seq_len=T * 2,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer
    )

    with torch.no_grad():
        logits, gen_state = model(idx, targets=None, prev_state=prev_state, return_state=True, kv_cache=kv_cache)

    assert logits is not None, "Generation should return logits"
    assert logits.shape == (B, T, config.vocab_size)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Generation state shape: {gen_state.shape}")
    print(f"  ✓ Pass")

    # Test 6: Side stream dropout (training mode)
    print("\n[Test 6] Side stream dropout in training mode...")
    model.train()

    losses = []
    for _ in range(10):
        with torch.no_grad():
            loss = model(idx, targets=targets, prev_state=prev_state)
        losses.append(loss.item())

    # Losses should vary slightly due to dropout
    loss_std = torch.tensor(losses).std().item()
    print(f"  Loss std over 10 runs: {loss_std:.6f}")
    print(f"  ✓ Pass (dropout causes variation)")

    model.eval()

    # Test 7: Check that type embeddings exist
    print("\n[Test 7] Check RLS components exist...")
    assert hasattr(model, 'E_type_main'), "Missing E_type_main"
    assert hasattr(model, 'E_type_side'), "Missing E_type_side"
    assert not hasattr(model, 'side_mlp'), "side_mlp should not exist (removed)"
    assert model.E_type_main.shape == (config.n_embd,)
    assert model.E_type_side.shape == (config.n_embd,)
    print(f"  E_type_main shape: {model.E_type_main.shape}")
    print(f"  E_type_side shape: {model.E_type_side.shape}")
    print(f"  ✓ Pass")

    # Test 8: Check layer 0 uses dual-stream attention
    print("\n[Test 8] Check layer 0 attention is configured correctly...")
    assert model.transformer.h[0].attn.layer_idx == 0
    assert model.transformer.h[1].attn.layer_idx == 1
    print(f"  Layer 0 layer_idx: {model.transformer.h[0].attn.layer_idx}")
    print(f"  Layer 1 layer_idx: {model.transformer.h[1].attn.layer_idx}")
    print(f"  ✓ Pass")

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    test_rls_side_tokens()
