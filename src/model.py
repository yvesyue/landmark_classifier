import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import *

class MyModel(nn.Module):
    def __init__(self, n_classes, dropout):
        super(MyModel, self).__init__()

        # Define individual blocks
        self.block1 = self._build_block(3, 64, dropout)
        self.block2 = self._build_block(64, 128, dropout)
        self.block3 = self._build_block(128, 256, dropout)
        self.block4 = self._build_block(256, 512, dropout)
        self.block5 = self._build_block(512, 1024, dropout)
        self.block6 = self._build_block(1024, 2048, dropout)
        
        # Define 1x1 convolutions for channel adjustments
        self.adjust1 = nn.Conv2d(64, 128, kernel_size=1)
        self.adjust2 = nn.Conv2d(128, 256, kernel_size=1)
        self.adjust3 = nn.Conv2d(256, 512, kernel_size=1)
        self.adjust4 = nn.Conv2d(512, 1024, kernel_size=1)
        
        # Fully Connected Layer
        self.fc = nn.Linear(2048, n_classes)

    def _build_block(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2+F.interpolate(self.adjust1(x1),scale_factor=0.5,mode='bilinear',
                                          align_corners=False))
        x4 = self.block4(x3 + F.interpolate(self.adjust2(x2), scale_factor=0.5, mode='bilinear',
                                            align_corners=False))
        x5 = self.block5(x4 + F.interpolate(self.adjust3(x3), scale_factor=0.5, mode='bilinear',
                                            align_corners=False))
        x6 = self.block6(x5 + F.interpolate(self.adjust4(x4), scale_factor=0.5, mode='bilinear', 
                                            align_corners=False))
        
        x = F.adaptive_avg_pool2d(x6, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(n_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
