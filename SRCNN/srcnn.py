from torch import nn
from config import configs
import torch.nn.functional as F

class SRCNN(nn.Module):  # 搭建SRCNN 3层卷积模型，Conve2d（输入层数，输出层数，卷积核大小，步长，填充层）
    def __init__(self, configs):
        super(SRCNN, self).__init__()
        self.num_channels = configs.n_colors
        self.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, self.num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.scale = configs.scale


    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x