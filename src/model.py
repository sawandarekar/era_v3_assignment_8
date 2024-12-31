import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class CIFAR10_Model(nn.Module):
    def __init__(self):
        super(CIFAR10_Model, self).__init__()

        drop_out_value = 0.1

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop_out_value)
        ) # output_size = 32x32x24   RF=3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            DepthwiseSeparableConv(24, 48, kernel_size=3, stride=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(drop_out_value)
        ) # output_size = 15x15x48   RF=5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(drop_out_value),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(drop_out_value)
        ) # output_size = 7x7x96   RF=13

        self.convblock4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(drop_out_value)
        ) # output_size = 4x4x128   RF=29

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  # Will convert any input size to 1x1

        # Final 1x1 conv to get 10 channels
        self.convblock5 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1)
        ) # output_size = 1x1x10   RF=29

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10)  # Flatten only the channel dimension
        return F.log_softmax(x, dim=-1)