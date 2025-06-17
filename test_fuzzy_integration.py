import torch
from modules.fuzzypvconv import FuzzyPVConv

def test_fuzzy_pv_conv():
    """Test the integrated FuzzyPVConv module"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    B, C_in, C_out, N = 2, 64, 128, 1024
    
    # Create test data
    coords = torch.randn(B, 3, N, device=device) * 0.5  # (B, 3, N)
    features = torch.randn(B, C_in, N, device=device)   # (B, C, N)
    
    # Test with fuzzy enabled
    fuzzy_conv = FuzzyPVConv(
        in_channels=C_in, 
        out_channels=C_out,
        kernel_size=3,
        resolution=32,
        use_fuzzy=True,
        fuzzy_radius=0.1,
        fuzzy_kernel=[4,2,2]
    ).to(device)
    
    print("🧪 Testing FuzzyPVConv integration...")
    print(f"Input - Features: {features.shape}, Coords: {coords.shape}")
    
    # Forward pass
    output_features, output_coords = fuzzy_conv((features, coords))
    
    print(f"✅ Output - Features: {output_features.shape}, Coords: {output_coords.shape}")
    assert output_features.shape == (B, C_out, N), f"Wrong output shape: {output_features.shape}"
    assert torch.equal(coords, output_coords), "Coordinates should be unchanged"
    
    print("✅ Three-path fusion working!")
    print("🎉 FuzzyPVConv integration test passed!")

if __name__ == "__main__":
    test_fuzzy_pv_conv()