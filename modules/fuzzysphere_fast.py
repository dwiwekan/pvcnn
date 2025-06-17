import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FastFuzzySphere(nn.Module):
    def __init__(self, kernel_size=[4, 2, 2], radius=0.05):
        super().__init__()
        self.n_azimuth, self.n_elevation, self.n_radial = kernel_size
        self.radius = radius
        self.total_bins = self.n_azimuth * self.n_elevation * self.n_radial

    def fuzzy_spherical_kernel_fast(self, database, query, nn_index, nn_count, nn_dist):
        """Optimized version with reduced memory allocation"""
        B, M, K = nn_index.shape
        device = database.device
        
        # Vectorized neighbor gathering - more efficient
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, K)
        neighbors = database[batch_idx, nn_index]  # More efficient than gather
        
        # Compute relative positions
        relative_pos = neighbors - query.unsqueeze(2)
        x, y, z = relative_pos.unbind(-1)
        
        # Vectorized spherical coordinates
        azimuth = torch.atan2(y, x) + math.pi
        elevation = torch.acos(torch.clamp(z / (nn_dist + 1e-8), -1, 1))
        radial = nn_dist / self.radius
        
        # Bin coordinates
        azimuth_bin = azimuth * (self.n_azimuth / (2 * math.pi))
        elevation_bin = elevation * (self.n_elevation / math.pi)  
        radial_bin = torch.clamp(radial, 0, self.n_radial - 1e-6)
        
        # Fast trilinear interpolation - vectorized
        a_floor = torch.floor(azimuth_bin).long()
        e_floor = torch.floor(elevation_bin).long()
        r_floor = torch.floor(radial_bin).long()
        
        # Fractional parts
        a_frac = azimuth_bin - a_floor.float()
        e_frac = elevation_bin - e_floor.float()
        r_frac = radial_bin - r_floor.float()
        
        # 8 corner weights (vectorized)
        weights = torch.stack([
            (1-a_frac) * (1-e_frac) * (1-r_frac),  # 000
            a_frac * (1-e_frac) * (1-r_frac),      # 100
            (1-a_frac) * e_frac * (1-r_frac),      # 010
            a_frac * e_frac * (1-r_frac),          # 110
            (1-a_frac) * (1-e_frac) * r_frac,      # 001
            a_frac * (1-e_frac) * r_frac,          # 101
            (1-a_frac) * e_frac * r_frac,          # 011
            a_frac * e_frac * r_frac               # 111
        ], dim=-1)  # (B, M, K, 8)
        
        # 8 corner indices (vectorized)
        indices = torch.stack([
            ((a_floor % self.n_azimuth) * self.n_elevation + 
             torch.clamp(e_floor, 0, self.n_elevation-1)) * self.n_radial + 
             torch.clamp(r_floor, 0, self.n_radial-1),
            (((a_floor + 1) % self.n_azimuth) * self.n_elevation + 
             torch.clamp(e_floor, 0, self.n_elevation-1)) * self.n_radial + 
             torch.clamp(r_floor, 0, self.n_radial-1),
            ((a_floor % self.n_azimuth) * self.n_elevation + 
             torch.clamp(e_floor + 1, 0, self.n_elevation-1)) * self.n_radial + 
             torch.clamp(r_floor, 0, self.n_radial-1),
            (((a_floor + 1) % self.n_azimuth) * self.n_elevation + 
             torch.clamp(e_floor + 1, 0, self.n_elevation-1)) * self.n_radial + 
             torch.clamp(r_floor, 0, self.n_radial-1),
            ((a_floor % self.n_azimuth) * self.n_elevation + 
             torch.clamp(e_floor, 0, self.n_elevation-1)) * self.n_radial + 
             torch.clamp(r_floor + 1, 0, self.n_radial-1),
            (((a_floor + 1) % self.n_azimuth) * self.n_elevation + 
             torch.clamp(e_floor, 0, self.n_elevation-1)) * self.n_radial + 
             torch.clamp(r_floor + 1, 0, self.n_radial-1),
            ((a_floor % self.n_azimuth) * self.n_elevation + 
             torch.clamp(e_floor + 1, 0, self.n_elevation-1)) * self.n_radial + 
             torch.clamp(r_floor + 1, 0, self.n_radial-1),
            (((a_floor + 1) % self.n_azimuth) * self.n_elevation + 
             torch.clamp(e_floor + 1, 0, self.n_elevation-1)) * self.n_radial + 
             torch.clamp(r_floor + 1, 0, self.n_radial-1)
        ], dim=-1)  # (B, M, K, 8)
        
        return indices, weights

class FastFuzzySphereMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[4, 2, 2], radius=0.05):
        super().__init__()
        self.fuzzy_sphere = FastFuzzySphere(kernel_size, radius)
        # Simplified filter weights
        self.filter_weights = nn.Parameter(torch.randn(self.fuzzy_sphere.total_bins, in_channels))
        nn.init.xavier_uniform_(self.filter_weights)
        
        if out_channels != in_channels:
            self.projection = nn.Linear(in_channels, out_channels)
        else:
            self.projection = None
            
    def forward(self, coords, features, nn_index, nn_count, nn_dist):
        B, N, C = features.shape
        _, M, K = nn_index.shape
        
        # Fast fuzzy kernel
        bin_index, bin_weights = self.fuzzy_sphere.fuzzy_spherical_kernel_fast(
            coords, coords, nn_index, nn_count, nn_dist)
        
        # Efficient neighbor feature gathering
        batch_idx = torch.arange(B, device=features.device).view(B, 1, 1).expand(B, M, K)
        neighbor_features = features[batch_idx, nn_index]  # (B, M, K, C)
        
        # Fast convolution - vectorized over all 8 bins
        output = torch.zeros(B, M, C, device=features.device)
        for i in range(8):
            weights = self.filter_weights[bin_index[..., i]]  # (B, M, K, C)
            weighted = neighbor_features * weights * bin_weights[..., i:i+1]
            output += weighted.sum(dim=2)  # Sum over neighbors
        
        if self.projection:
            output = self.projection(output)
            
        return output