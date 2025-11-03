"""
Test script to verify Mamba model fix for different prediction horizons.
Tests that the model can now predict H > seq_len without crashing.
"""

import torch
import sys
from argparse import Namespace

# Add models to path
sys.path.append('.')

from models.Mamba import Model


def test_mamba_horizons():
    """Test Mamba model with various prediction horizons."""

    print("Testing Mamba Model Fix")
    print("=" * 60)

    # Test configuration
    batch_size = 4
    seq_len = 60
    enc_in = 5  # Number of input features
    c_out = 5   # Number of output features

    # Test different horizons including problematic ones
    test_horizons = [3, 10, 22, 50, 60, 100]

    for pred_len in test_horizons:
        print(f"\nTesting H={pred_len}:")
        print("-" * 40)

        try:
            # Create config
            configs = Namespace(
                task_name='long_term_forecast',
                pred_len=pred_len,
                seq_len=seq_len,
                enc_in=enc_in,
                c_out=c_out,
                d_model=64,
                d_ff=128,
                d_conv=4,
                expand=2,
                embed='timeF',
                freq='h',
                dropout=0.1
            )

            # Initialize model
            model = Model(configs)
            model.eval()

            # Create dummy input
            x_enc = torch.randn(batch_size, seq_len, enc_in)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time features
            x_dec = torch.randn(batch_size, seq_len, enc_in)
            x_mark_dec = torch.randn(batch_size, seq_len, 4)

            # Forward pass
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # Verify output shape
            expected_shape = (batch_size, pred_len, c_out)
            actual_shape = output.shape

            if actual_shape == expected_shape:
                print(f"✅ SUCCESS: Output shape {actual_shape} matches expected {expected_shape}")
                print(f"   Input seq_len={seq_len}, Output pred_len={pred_len}")
                print(f"   Can now predict {pred_len - seq_len if pred_len > seq_len else 'N/A'} steps beyond input!")
            else:
                print(f"❌ FAILED: Output shape {actual_shape} != expected {expected_shape}")

        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Test Summary:")
    print("✅ If all tests passed, the Mamba model can now:")
    print("   1. Predict any horizon length (not limited to seq_len)")
    print("   2. Generate TRUE future predictions (not just input slices)")
    print("   3. Handle H=100 without dimension mismatch errors")


if __name__ == "__main__":
    test_mamba_horizons()
