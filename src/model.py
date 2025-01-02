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

        drop_out_value = 0.02

        # Input Block # Input: 32x32x3  # convblock1: Conv 3x3, s=1, p=1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(drop_out_value)
        ) 
        # Nout = (Nin + 2P - k)/S + 1 = (32 + 2*1 - 3)/1 + 1 = 32
        # Rout = Rin + (k-1)*Jin = 1 + (3-1)*1 = 3
        # Jout = Jin*S = 1*1 = 1

        # Nin=32 Rin=1 Jin=1 S=1 P=1 Nout=32 Rout=3 Jout=1


        # CONVOLUTION BLOCK 1: DepthwiseSeparableConv 3x3, s=2, p=1
        self.convblock2 = nn.Sequential(
            DepthwiseSeparableConv(24, 48, kernel_size=3, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(drop_out_value)
        )
        # Nout = (32 + 2*1 - 3)/2 + 1 = 16
        # Rout = 3 + (3-1)*1 = 5
        # Jout = 1*2 = 2

        #Nin=32   Rin=3   Jin=1  S=2  P=1  Nout=16  Rout=5  Jout=2

        self.convblock3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=2, dilation=2),
            # convblock3: First Conv (Dilated) 3x3, s=1, p=2, d=2

            # Nout = (16 + 2*2 - 2*(3-1))/1 + 1 = 16
            # Rout = 5 + (3-1)*2*2 = 9 (dilated conv multiplies by dilation)
            # Jout = 2*1 = 2

            # First Conv: Nin=16 Rin=5 Jin=2 S=1 P=2 D=2 Nout=16 Rout=9 Jout=2


            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(drop_out_value),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            # convblock3: Second Conv 3x3, s=2, p=1

            # Nout = (16 + 2*1 - 3)/2 + 1 = 8
            # Rout = 9 + (3-1)*2 = 13
            # Jout = 2*2 = 4

            #Nin=16   Rin=9   Jin=2  S=2      P=1  Nout=8   Rout=13 Jout=4

            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(drop_out_value)
        ) 
        # Second Conv: Nin=16 Rin=9 Jin=2 S=2 P=1 Nout=8 Rout=13 Jout=4

        # convblock4: Conv 3x3, s=2, p=1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(drop_out_value)
        ) # Nin=8 Rin=13 Jin=4 S=2 P=1 Nout=4 Rout=21 Jout=8

        # Nout = (8 + 2*1 - 3)/2 + 1 = 4
        # Rout = 13 + (3-1)*4 = 21
        # Jout = 4*2 = 8

        #Nin=8    Rin=13  Jin=4  S=2      P=1  Nout=4   Rout=21 Jout=8

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)  
        # Nin=4 Rin=21 Jin=8 Nout=1 Rout=45 Jout=8

        # Final 1x1 conv to get 10 channels
        self.convblock5 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=1)
        ) # Nin=1 Rin=45 Jin=8 S=1 P=0 Nout=1 Rout=45 Jout=8

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.convblock5(x)
        x = x.view(-1, 10)  # Flatten only the channel dimension
        return F.log_softmax(x, dim=-1)