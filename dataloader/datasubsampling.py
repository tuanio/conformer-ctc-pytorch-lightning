import torch
from torch import nn


class DataSubsampling(nn.Module):
    """
    this class will downsample data
    """

    def __init__(self, kernel_size):
        super(DataSubsampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


data_subsampling = DataSubsampling(kernel_size=(2, 2))
