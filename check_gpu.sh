#!/bin/bash

# GPU diagnostics script - checks if CUDA is properly configured

echo "=========================================="
echo "GPU/CUDA Diagnostics"
echo "=========================================="
echo ""

echo "1. Checking for NVIDIA GPU..."
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
else
    echo "❌ nvidia-smi not found - no NVIDIA GPU detected"
    echo ""
fi

echo "2. Checking PyTorch CUDA availability..."
echo "----------------------------------------"
source .venv/bin/activate 2>/dev/null || true

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print('✅ CUDA is working!')
else:
    print('❌ CUDA not available - PyTorch will use CPU')
    print('')
    print('To fix this, run:')
    print('  pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
"
echo ""

echo "3. Checking what device training will use..."
echo "----------------------------------------"
python -c "
from nanochat.common import autodetect_device_type
device = autodetect_device_type()
print(f'Training will use: {device}')
"
echo ""

echo "=========================================="
echo "Diagnostics complete"
echo "=========================================="
