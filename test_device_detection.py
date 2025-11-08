#!/usr/bin/env python3
"""
Test script to verify device detection works correctly.
"""

import torch
import sys
import os
import json

# Add the ddpm_pretraining directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ddpm_pretraining'))

def test_device_detection():
    """Test device detection logic."""
    print("PyTorch version:", torch.__version__)
    print("Python version:", sys.version)
    
    # Simulate config
    config = {
        "gpu": 0,
        "experiment_path": "ddpm_pretraining/ddpm_pretraining_experiments",
    }
    
    # Check for available devices (CUDA for NVIDIA, MPS for Apple Silicon, CPU as fallback)
    if torch.cuda.is_available():
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            GPU = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            GPU = config["gpu"]
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
        device = torch.device("cuda")
        print(f"Torch GPU Name: {torch.cuda.get_device_name(0)}... Using GPU {GPU}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Torch MPS available... Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Torch GPU not available... Using CPU")
        
    print(f"Selected device: {device}")
    print(f"Device type: {device.type}")
    
    # Test tensor operations on the selected device
    try:
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        print(f"✅ Tensor operations working correctly on {device}")
        print(f"Tensor device: {z.device}")
    except Exception as e:
        print(f"❌ Error during tensor operations: {e}")

if __name__ == "__main__":
    test_device_detection()
