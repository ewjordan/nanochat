# RLS Development Files

This directory contains development artifacts for the Recurrent Layer State (RLS) feature.

## Directory Structure

```
rls_dev/
├── docs/           # Documentation and validation summaries
├── tests/          # Test scripts for debugging RLS implementation
└── scripts/        # Training and validation scripts
```

## Contents

### `docs/`
- `LOCAL_TRAINING_GUIDE.md` - Guide for running RLS training on M4 Mac
- `LOCAL_VALIDATION.md` - Validation procedures for RLS feature
- `VALIDATION_SUMMARY.md` - Summary of validation results

### `tests/`
Test scripts used during RLS development to debug specific issues:
- `test_bfloat16_issue.py` - Debugging bfloat16 precision issues
- `test_gradient_simple.py` - Testing gradient flow
- `test_rls_activations.py` - Testing RLS activation patterns
- `test_state_gate_init.py` - Testing state gate initialization
- `test_training_steps.py` - Testing individual training steps

### `scripts/`
- `local_real_train.sh` - Local training script for M4 Mac validation
- `local_validation_run.py` - Validation runner script

## Related Files

**Production scripts** (in project root):
- `local_train_rls_full.sh` - Full RLS training comparison script (A100)

**Issue documentation** (in project root):
- `TORCH_COMPILE_GRADIENT_ISSUE.md` - Documents torch.compile gradient issue for upstream filing

## Experiment Outputs

Experiment logs and checkpoints are written to:
- `local_rls_experiments/` - M4 Mac experiment outputs (gitignored)
- `local_rls_experiments_full/` - A100 experiment outputs (gitignored)

These directories contain:
- Training logs (baseline.log, rls.log)
- Model checkpoints
- Validation metrics
- Sample outputs

## Usage

See individual documentation files in `docs/` for usage instructions.
