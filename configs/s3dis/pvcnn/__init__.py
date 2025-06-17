import torch.optim as optim

from models.s3dis import PVCNN
from utils.config import Config, configs

# model
configs.model = Config(PVCNN)
configs.model.num_classes = configs.data.num_classes
configs.model.extra_feature_channels = 6
configs.model.dilation_rates = [1, 1, 1, 1]  # CONSERVATIVE: No dilation first
# configs.model.use_fuzzy = False  # Disable fuzzy for baseline test

configs.dataset.num_points = 4096

# Optimizer settings
configs.train.optimizer.weight_decay = 1e-4  # Increase weight decay

# train: scheduler  
configs.train.scheduler = Config(optim.lr_scheduler.CosineAnnealingLR)
configs.train.scheduler.T_max = configs.train.num_epochs