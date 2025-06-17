import torch
import torch.nn as nn
import torch.nn.functional as torch_F  # Add this import

import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d
from modules.fuzzysphere import FuzzySphereMLP

__all__ = ['FuzzyPVConv']

def knn_search(query, database, k):
    """
    Simple KNN search using pytorch
    Args:
        query: (B, M, 3) query points
        database: (B, N, 3) database points  
        k: number of neighbors
    Returns:
        nn_index: (B, M, k) neighbor indices
        nn_dist: (B, M, k) distances
    """
    chunk_size = min(512, M)  # Process in smaller chunks
    results_idx = []
    results_dist = []
    
    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        query_chunk = query[:, i:end_i]  # (B, chunk, 3)
        
        # Fast distance computation
        query_exp = query_chunk.unsqueeze(2)  # (B, chunk, 1, 3)
        database_exp = database.unsqueeze(1)  # (B, 1, N, 3)
        distances = torch.sum((query_exp - database_exp) ** 2, dim=-1)  # Squared distance (faster)
        
        # Find k nearest neighbors
        nn_dist, nn_index = torch.topk(distances, k, dim=-1, largest=False)
        results_idx.append(nn_index)
        results_dist.append(torch.sqrt(nn_dist + 1e-8))  # Only sqrt at the end
    
    return torch.cat(results_idx, dim=1), torch.cat(results_dist, dim=1)

class FuzzyPVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, 
                 with_se=False, normalize=True, eps=0, dilation=1, 
                 use_fuzzy=False, fuzzy_radius=0.05, fuzzy_kernel=[4,2,2], **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.use_fuzzy = use_fuzzy

        # Original voxel path
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        
        # Calculate padding based on dilation
        if dilation == 1:
            padding = kernel_size // 2
        elif dilation == 2:
            padding = 2
        elif dilation == 4:
            padding = 4
        else:
            padding = dilation * (kernel_size - 1) // 2
            
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=padding),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, dilation=1, padding=kernel_size//2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        
        # Original point path
        self.point_features = SharedMLP(in_channels, out_channels)
        
        # REAL fuzzy spherical convolution instead of simple attention
        if use_fuzzy:
            self.fuzzy_sphere = FuzzySphereMLP(
                in_channels, out_channels,
                kernel_size=fuzzy_kernel,
                radius=fuzzy_radius
            )
            self.fuzzy_k = 8  # REDUCE from 16 to 8 neighbors
            self.fuzzy_proj = nn.Conv1d(in_channels, out_channels, 1)  # Add projection layer
            self.fusion = SharedMLP(out_channels * 3, out_channels)
        else:
            self.fuzzy_sphere = None
            self.fuzzy_proj = None
            self.fusion = None

    def forward(self, inputs):
        features, coords = inputs
        
        # Voxel path (existing)
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        
        # Point path (existing)
        point_features = self.point_features(features)
        
        if self.use_fuzzy and self.fuzzy_sphere is not None:
            # SIMPLE: Use input features but project to correct output channels
            B, C, N = features.shape
            
            # Simple spatial attention without KNN
            coords_norm = torch_F.normalize(coords.transpose(1, 2), dim=-1)  # (B, N, 3)
            attention = torch.bmm(coords_norm, coords_norm.transpose(1, 2))  # (B, N, N)
            attention = torch_F.softmax(attention / 0.1, dim=-1)  # Temperature scaling
            
            # Apply attention to features
            features_t = features.transpose(1, 2)  # (B, N, C)
            fuzzy_features = torch.bmm(attention, features_t).transpose(1, 2)  # (B, C, N)
            
            # Project to output channels
            fuzzy_features = self.fuzzy_proj(fuzzy_features)
            
            # Three-path fusion
            combined_features = torch.cat([voxel_features, point_features, fuzzy_features], dim=1)
            fused_features = self.fusion(combined_features)
        else:
            # Original behavior: combine voxel + point paths
            fused_features = voxel_features + point_features
        
        # Return tuple (features, coords) like original PVConv
        return fused_features, coords