from torch import nn
from config import configs
import torch.nn.functional as F
import math

def normNN(N, d, k, p=0):
    NN = nn.Conv2d(N, d, kernel_size=k, padding=p)
    std = math.sqrt(2/d/NN.weight.data[0][0].numel())
    nn.init.normal_(NN.weight.data, mean=0.0, std=std)
    nn.init.zeros_(NN.bias.data)
    return NN

class FSRCNN(nn.Module):
    def __init__(self, configs):
        super(FSRCNN, self).__init__()
        scale_factor = configs.scale
        N = 3
        d = 56
        s = 12
        m = 4
        # 第一组
        nns = [normNN(N, d, 5, 5//2), nn.PReLU(d)]
        nns += [normNN(d, s, 1), nn.PReLU(s)]   #降维
        for _ in range(m):      #匹配
            nns += [normNN(s, s, 3, 3//2), nn.PReLU(s)]
        nns += [nn.Conv2d(s, d, 1), nn.PReLU(d)]    #放大
        self.NN1 = nn.Sequential(*nns)
        # 反卷积层
        self.NN2 = nn.ConvTranspose2d(d, N, 9, scale_factor,
            9//2, output_padding=scale_factor-1)
        nn.init.normal_(self.NN2.weight.data,
            mean=0.0, std=0.001)
        nn.init.zeros_(self.NN2.bias.data)
    def forward(self, x):
        x = self.NN1(x)
        x = self.NN2(x)
        return x
