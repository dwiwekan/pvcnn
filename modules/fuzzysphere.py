import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FuzzySphere(nn.Module):
    def __init__(self, kernel_size=[4, 2, 2], radius=0.05):
        super().__init__()
        self.n_azimuth, self.n_elevation, self.n_radial = kernel_size
        self.radius = radius
        self.total_bins = self.n_azimuth * self.n_elevation * self.n_radial
        
        # OPTIMIZATION: Precompute constants
        self.azimuth_scale = self.n_azimuth / (2 * math.pi)
        self.elevation_scale = self.n_elevation / math.pi
        
    def fuzzy_spherical_kernel_pt(self, database, query, nn_index, nn_count, nn_dist):
        """ULTRA-OPTIMIZED fuzzy spherical kernel"""
        B, M, K = nn_index.shape
        device = database.device
        
        # OPTIMIZATION: Use index_select instead of advanced indexing (faster)
        neighbors = database[torch.arange(B, device=device).view(B, 1, 1).expand(B, M, K), nn_index]
        
        # Compute relative positions (vectorized)
        relative_pos = neighbors - query.unsqueeze(2)
        x, y, z = relative_pos.unbind(-1)
        
        # Fast spherical coordinates with precomputed scales
        azimuth = torch.atan2(y, x) + math.pi
        elevation = torch.acos(torch.clamp(z / (nn_dist + 1e-8), -1, 1))
        
        # Use precomputed scales
        azimuth_bin = azimuth * self.azimuth_scale
        elevation_bin = elevation * self.elevation_scale
        radial_bin = torch.clamp(nn_dist / self.radius, 0, self.n_radial - 1e-6)
        
        # OPTIMIZATION: Combine floor and frac operations
        a_floor = azimuth_bin.floor()
        e_floor = elevation_bin.floor()
        r_floor = radial_bin.floor()
        
        a_frac = azimuth_bin - a_floor
        e_frac = elevation_bin - e_floor  
        r_frac = radial_bin - r_floor
        
        # Convert to long once
        a_floor = a_floor.long()
        e_floor = e_floor.long()
        r_floor = r_floor.long()
        
        # OPTIMIZATION: Precompute 1-frac values
        a_frac_inv = 1 - a_frac
        e_frac_inv = 1 - e_frac
        r_frac_inv = 1 - r_frac
        
        # Vectorized trilinear weights (most optimized)
        bin_coeffs = torch.stack([
            a_frac_inv * e_frac_inv * r_frac_inv,  # 000
            a_frac * e_frac_inv * r_frac_inv,      # 100
            a_frac_inv * e_frac * r_frac_inv,      # 010
            a_frac * e_frac * r_frac_inv,          # 110
            a_frac_inv * e_frac_inv * r_frac,      # 001
            a_frac * e_frac_inv * r_frac,          # 101
            a_frac_inv * e_frac * r_frac,          # 011
            a_frac * e_frac * r_frac               # 111
        ], dim=-1)  # (B, M, K, 8)
        
        # OPTIMIZATION: Batch clamp operations
        e_clamped = torch.clamp(e_floor, 0, self.n_elevation-1)
        e_clamped_p1 = torch.clamp(e_floor + 1, 0, self.n_elevation-1)
        r_clamped = torch.clamp(r_floor, 0, self.n_radial-1)
        r_clamped_p1 = torch.clamp(r_floor + 1, 0, self.n_radial-1)
        
        # Precompute base indices
        base_a = (a_floor % self.n_azimuth) * self.n_elevation
        base_a_p1 = ((a_floor + 1) % self.n_azimuth) * self.n_elevation
        
        # Vectorized bin indices (optimized)
        bin_indices = torch.stack([
            (base_a + e_clamped) * self.n_radial + r_clamped,
            (base_a_p1 + e_clamped) * self.n_radial + r_clamped,
            (base_a + e_clamped_p1) * self.n_radial + r_clamped,
            (base_a_p1 + e_clamped_p1) * self.n_radial + r_clamped,
            (base_a + e_clamped) * self.n_radial + r_clamped_p1,
            (base_a_p1 + e_clamped) * self.n_radial + r_clamped_p1,
            (base_a + e_clamped_p1) * self.n_radial + r_clamped_p1,
            (base_a_p1 + e_clamped_p1) * self.n_radial + r_clamped_p1
        ], dim=-1)  # (B, M, K, 8)
        
        return bin_indices, bin_coeffs

    def fuzzy_depthwise_conv3d_pt(self, input_features, filter_weights, nn_index, nn_count, bin_index, bin_coeff):
        """ULTRA-OPTIMIZED fuzzy convolution"""
        B, M, K = nn_index.shape
        _, N, C_in = input_features.shape
        
        # OPTIMIZATION: Single gather operation
        batch_idx = torch.arange(B, device=input_features.device).view(B, 1, 1).expand(B, M, K)
        neighbor_features = input_features[batch_idx, nn_index]  # (B, M, K, C_in)
        
        # OPTIMIZATION: More efficient weight gathering and computation
        weights = filter_weights[bin_index]  # (B, M, K, 8, C_in, 1)
        
        # OPTIMIZATION: Combine operations to reduce memory allocation
        weighted = (neighbor_features.unsqueeze(-2).unsqueeze(-1) * 
                   weights * 
                   bin_coeff.unsqueeze(-1).unsqueeze(-1))  # (B, M, K, 8, C_in, 1)
        
        # Single reduction operation
        output = weighted.sum(dim=(2, 3)).squeeze(-1)  # (B, M, C_in)
        
        return output

class FuzzySphereMLP(nn.Module):
    """Fuzzy sphere convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size=[4, 2, 2], radius=0.05):
        super().__init__()
        self.fuzzy_sphere = FuzzySphere(kernel_size, radius)
        # Reduced filter complexity
        self.filter_weights = nn.Parameter(torch.randn(self.fuzzy_sphere.total_bins, in_channels, 1))
        nn.init.xavier_uniform_(self.filter_weights)
        
        if out_channels != in_channels:
            self.projection = nn.Linear(in_channels, out_channels)
        else:
            self.projection = None
            
    def forward(self, coords, features, nn_index, nn_count, nn_dist):
        # Apply fuzzy sphere convolution
        bin_index, bin_coeff = self.fuzzy_sphere.fuzzy_spherical_kernel_pt(
            coords, coords, nn_index, nn_count, nn_dist)
            
        fuzzy_features = self.fuzzy_sphere.fuzzy_depthwise_conv3d_pt(
            features, self.filter_weights, nn_index, nn_count, bin_index, bin_coeff)
            
        if self.projection:
            fuzzy_features = self.projection(fuzzy_features)
            
        return fuzzy_features

# Add this for easy import
__all__ = ['FuzzySphere', 'FuzzySphereMLP']

