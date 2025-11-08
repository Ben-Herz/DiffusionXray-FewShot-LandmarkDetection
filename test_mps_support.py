#!/usr/bin/env python3
"""
Test script to verify MPS (Metal Performance Shaders) support on Apple Silicon Macs.
"""

import torch
import sys

def test_mps_support():
    """Test MPS support and print system information."""
    print("PyTorch version:", torch.__version__)
    print("Python version:", sys.version)
    
    # Check for MPS support
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available")
        device = torch.device("mps")
        
        # Test tensor operations on MPS
        try:
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print("✅ MPS tensor operations working correctly")
            print(f"Tensor device: {z.device}")
        except Exception as e:
            print(f"❌ Error during MPS operations: {e}")
    else:
        print("⚠️  MPS (Metal Performance Shaders) is not available")
        print("   This could be because:")
        print("   - You're not on an Apple Silicon Mac")
        print("   - Your macOS version is too old (requires macOS 12.3+)")
        print("   - Your PyTorch version doesn't support MPS")
        
    # Check CPU fallback
    print("\nFallback information:")
    device = torch.device("cpu")
    print(f"CPU device: {device}")
    
    # Test basic tensor operations
    try:
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print("✅ CPU tensor operations working correctly")
    except Exception as e:
        print(f"❌ Error during CPU operations: {e}")

if __name__ == "__main__":
    test_mps_support()
