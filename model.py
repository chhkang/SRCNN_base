import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock


class SRCNN(nn.Module):
    def __init__(self,upscale_factor):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 96, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(96, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.relu6(self.conv1(x))
        x = F.relu6(self.conv2(x))
        x = F.relu6(self.conv3(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv4(x)))
        return x