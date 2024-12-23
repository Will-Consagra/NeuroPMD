import torch
from torch.utils.data import Dataset

class PointPattern(Dataset):
    """Point cloud representation of point pattern of a D-dimensional product p-manifold 
    Args:
        point_pattern (torch.tensor): num_points x p * D
    """
    def __init__(self, point_pattern):
        self.point_pattern = point_pattern
    def __getitem__(self, index):
        return self.point_pattern[index,...]
    def __len__(self):
        return self.point_pattern.shape[0]