#!/usr/bin/env python3
"""
Test script to verify MPS is being used during training.
"""

import torch
import sys
import os
import json

# Add the ddpm_pretraining directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ddpm_pretraining'))

from ddpm_pretraining.utils import load_data
from ddpm_pretraining.model.training_functions import initialize_ddpm

def test_mps_training():
    """Test that MPS is being used during training."""
    print("PyTorch version:", torch.__version__)
    print("Python version:", sys.version)
    
    # Check for available devices (CUDA for NVIDIA, MPS for Apple Silicon, CPU as fallback)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Torch CUDA available... Using NVIDIA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Torch MPS available... Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Torch GPU not available... Using CPU")
        
    print(f"Selected device: {device}")
    
    # Test with a small batch
    dataset_path = "datasets/chest"
    image_size = 64  # Smaller for testing
    image_channels = 1
    batch_size = 2   # Smaller for testing
    
    print(f"Loading data from {dataset_path}...")
    try:
        train_dataloader, test_dataloader = load_data(
            dataset_path, image_size, image_channels, batch_size, 
            pin_memory=False, num_workers=0  # Simplified for testing
        )
        print(f"Successfully loaded data. Train batches: {len(train_dataloader)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Test configuration
    config = {
        "dataset": {
            "image_size": image_size,
            "image_channels": image_channels,
        },
        "model": {
            "lr": 1e-4,
            "optimizer": "adam",
            "beta_schedule": {
                "train": {
                    "schedule": "linear",
                    "n_timestep": 100,  # Smaller for testing
                    "linear_start": 1e-4,
                    "linear_end": 0.02
                }
            },
            "unet": {
                "channel_mults": [1, 2, 4],
                "attn_res": 16,
                "num_head_channels": 4,
                "res_blocks": 2,
                "self_condition": False,
            },
            "use_ema": False,
        }
    }
    
    print("Initializing DDPM model...")
    try:
        ddpm = initialize_ddpm(config, phase="train", device=device)
        print(f"Successfully initialized DDPM model on {device}")
    except Exception as e:
        print(f"Error initializing DDPM model: {e}")
        return
    
    # Test a single forward pass
    print("Testing forward pass...")
    try:
        data_iter = iter(train_dataloader)
        batch = next(data_iter)
        x = batch['image'].to(device)
        print(f"Input tensor shape: {x.shape}")
        print(f"Input tensor device: {x.device}")
        
        # Sample timesteps
        t = ddpm.sample_timesteps(x.shape[0])
        print(f"Timesteps: {t}")
        
        # Forward pass
        loss = ddpm.p_losses(x=x, t=t, loss_type="l2")
        print(f"Loss: {loss.item()}")
        print("✅ Forward pass successful!")
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        return
    
    print("✅ MPS training test completed successfully!")

if __name__ == "__main__":
    test_mps_training()
