import torch.nn as nn

import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d

__all__ = ['PVConv']


# ...existing code...
class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        
        # Calculate padding based on dilation
        if dilation == 1:
            padding = kernel_size // 2
        elif dilation == 2:
            padding = 2  # For kernel_size=3, dilation=2
        elif dilation == 4:
            padding = 4  # For kernel_size=3, dilation=4
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
        self.point_features = SharedMLP(in_channels, out_channels)
    
    # ...existing forward method...

    def forward(self, inputs):
        features, coords = inputs
        
        # For now, just do original PVConv behavior to test integration
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        
        point_features = self.point_features(features)
        fused_features = voxel_features + point_features
        
        # CRITICAL: Return tuple like original PVConv
        return fused_features, coords
