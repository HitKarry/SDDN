import common

import torch.nn as nn

def make_model(args, parent=False):
    return MDSR(args)

class MDSR(nn.Module):
    def __init__(self, configs, conv=common.default_conv):
        super(MDSR, self).__init__()
        n_resblocks = configs.n_resblocks
        n_feats = configs.n_feats
        kernel_size = 3
        self.scale_idx = 0

        act = nn.ReLU(True)

        rgb_mean = (4.591007e-07, -1.584896e-06, 8.482833e-06)
        rgb_std = (0.9999993, 0.99999946, 0.9999999)
        self.sub_mean = common.MeanShift(configs.rgb_range, rgb_mean, rgb_std)

        m_head = [conv(configs.n_colors, n_feats, kernel_size)]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in configs.scale
        ])

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.upsample = nn.ModuleList([
            common.Upsampler(
                conv, s, n_feats, act=False
            ) for s in configs.scale
        ])

        m_tail = [conv(n_feats, configs.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(configs.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        x = self.upsample[self.scale_idx](res)
        x = self.tail(x)
        # x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

