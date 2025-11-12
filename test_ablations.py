#!/usr/bin/env python3
"""
Test RLS ablation flags to verify they work correctly.

Tests:
1. Normal RLS (baseline)
2. zero_prev_state ablation
3. mask_side_attention ablation

Verifies that E_type gradients behave as expected under each condition.
"""

import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader

def test_ablation(config_name, config):
    """Test a single configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # Create model
    model = GPT(config).to(device)
    model.train()

    # Get data
    tokenizer = get_tokenizer()
    data_loader = tokenizing_distributed_data_loader(B=1, T=512, split="train", tokenizer_threads=1, device=device)
    x, y = next(data_loader)

    # Forward + backward
    model.zero_grad()
    loss = model.forward_with_recurrence(x, y)
    loss.backward()

    # Check gradients
    E_main_grad = model.E_type_main.grad.norm().item() if model.E_type_main.grad is not None else 0.0
    E_side_grad = model.E_type_side.grad.norm().item() if model.E_type_side.grad is not None else 0.0
    ratio = E_side_grad / max(E_main_grad, 1e-10)

    print(f"Loss: {loss.item():.6f}")
    print(f"E_type_main grad: {E_main_grad:.6f}")
    print(f"E_type_side grad: {E_side_grad:.6f}")
    print(f"Ratio (side/main): {ratio:.2f}x")

    # Interpretation
    if E_side_grad < 1e-6:
        print("✓ Side stream effectively disabled (near-zero gradients)")
    elif ratio > 10:
        print("⚠️  Side stream dominates (>10x gradient)")
    else:
        print("✓ Side and main streams balanced")

    return E_main_grad, E_side_grad, ratio

if __name__ == "__main__":
    print("="*60)
    print("RLS Ablation Testing")
    print("="*60)

    base_config = {
        'vocab_size': 65536,
        'n_layer': 12,
        'n_embd': 768,
        'n_head': 6,
        'n_kv_head': 6,
        'sequence_len': 512,
        'recurrent_layer_state': True,
        'num_recurrence_warmup': 1,
    }

    # Test 1: Normal RLS (expect side >> main)
    config1 = GPTConfig(**base_config)
    main1, side1, ratio1 = test_ablation("Normal RLS (baseline)", config1)

    # Test 2: Zero prev_state (expect side ~= 0)
    config2 = GPTConfig(**base_config, zero_prev_state=True)
    main2, side2, ratio2 = test_ablation("RLS with zero_prev_state", config2)

    # Test 3: Mask side attention (expect side ~= 0)
    config3 = GPTConfig(**base_config, mask_side_attention=True)
    main3, side3, ratio3 = test_ablation("RLS with mask_side_attention", config3)

    # Test 4: 100% side dropout (expect side ~= 0)
    config4 = GPTConfig(**base_config, side_dropout_rate=1.0)
    main4, side4, ratio4 = test_ablation("RLS with 100% side dropout", config4)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Configuration':<30} {'E_main':>12} {'E_side':>12} {'Ratio':>10}")
    print("-"*60)
    print(f"{'Normal RLS':<30} {main1:>12.6f} {side1:>12.6f} {ratio1:>10.2f}x")
    print(f"{'zero_prev_state':<30} {main2:>12.6f} {side2:>12.6f} {ratio2:>10.2f}x")
    print(f"{'mask_side_attention':<30} {main3:>12.6f} {side3:>12.6f} {ratio3:>10.2f}x")
    print(f"{'100% side dropout':<30} {main4:>12.6f} {side4:>12.6f} {ratio4:>10.2f}x")

    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    # Validate that ablations work as expected
    all_pass = True

    if ratio1 < 5:
        print("❌ FAIL: Normal RLS should have side >> main (ratio > 5)")
        all_pass = False
    else:
        print(f"✓ Normal RLS shows side dominance ({ratio1:.1f}x)")

    if side2 > 0.01:
        print(f"❌ FAIL: zero_prev_state should nearly eliminate side gradients (got {side2:.6f})")
        all_pass = False
    else:
        print(f"✓ zero_prev_state disables side stream ({side2:.6f})")

    if side3 > 0.01:
        print(f"❌ FAIL: mask_side_attention should nearly eliminate side gradients (got {side3:.6f})")
        all_pass = False
    else:
        print(f"✓ mask_side_attention disables side stream ({side3:.6f})")

    if side4 > 0.01:
        print(f"❌ FAIL: 100% side dropout should nearly eliminate side gradients (got {side4:.6f})")
        all_pass = False
    else:
        print(f"✓ 100% side dropout disables side stream ({side4:.6f})")

    if all_pass:
        print("\n✅ All ablations working correctly!")
    else:
        print("\n❌ Some ablations not working as expected")
