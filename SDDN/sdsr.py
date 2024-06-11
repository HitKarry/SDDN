import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import configs
import numpy as np
import copy
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out

class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=10):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes//ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        maxout = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + maxout

        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def SelfChannelAttention(query, key, value, dropout=None):
    """
    attention over the two Channel axes
    Args:
        query, key, value: linearly-transformed query, key, value  (N, h, S, C, D)
        mask: None (space attention does not need mask), this argument is intentionally set for consistency
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)

def SelfSpaceAttention(query, key, value, dropout=None):
    """
    attention over the two space axes
    Args:
        query, key, value: linearly-transformed query, key, value (N, h, S, C, D)
        mask: None (Channel attention does not need mask), this argument is intentionally set for consistency
    """
    d_k = query.size(-1)
    query = query.transpose(2, 3)  # (N, h, C, S, D)
    key = key.transpose(2, 3)  # (N, h, C, S, D)
    value = value.transpose(2, 3)  # (N, h, C, S, D)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)  # (N, h, C, S, S)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).transpose(2, 3)

def SelfChannelSpaceAttention(query, key, value, dropout=None):
    """
    attention over the two Channel axes
    Args:
        query, key, value: linearly-transformed query, key, value  (N, h, S, C, D)
        mask: None (space attention does not need mask), this argument is intentionally set for consistency
    """
    N, h, S, C,d_k = query.size(0),query.size(1),query.size(2),query.size(3),query.size(-1)
    query, key, value = query.flatten(start_dim=2,end_dim=3), key.flatten(start_dim=2,end_dim=3), value.flatten(start_dim=2,end_dim=3)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).view(N, h, S, C, d_k)

def fold_tensor(tensor, output_size, kernel_size):
    """
    reconstruct the image from its non-overlapping patches
    Args:
        input tensor of size (N, *, C, k_h*k_w, n_h, n_w)
        output_size of size(H, W), the size of the original image to be reconstructed
        kernel_size: (k_h, k_w)
        stride is usually equal to kernel_size for non-overlapping sliding window
    Returns:
        (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    tensor = tensor.float()
    n_dim = len(tensor.size())
    assert n_dim == 4 or n_dim == 6
    f = tensor.flatten(0, 2) if n_dim == 6 else tensor
    folded = F.fold(f.flatten(-2), output_size=output_size, kernel_size=kernel_size, stride=kernel_size)
    if n_dim == 6:
        folded = folded.reshape(tensor.size(0), tensor.size(1), tensor.size(2), *folded.size()[1:]).squeeze(-3)
    return folded

def unfold_StackOverChannel(img, kernel_size=[2,2]):
    """
    divide the original image to patches, then stack the grids in each patch along the channels
    Args:
        img (N, *, C, H, W): the last two dimensions must be the spatial dimension
        kernel_size: tuple of length 2
    Returns:
        output (N, T_src, C, H_k*N_k, H_output, W_output)
    """
    n_dim = len(img.size())
    assert n_dim == 4

    pt = img.unfold(-2, size=kernel_size[0], step=kernel_size[0])
    pt = pt.unfold(-2, size=kernel_size[1], step=kernel_size[1]).flatten(-2)  # (N, C, n0, n1, k0*k1)
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3) # (N, C, H_k*N_k, H_output, W_output)
    assert pt.size(-3) == kernel_size[0] * kernel_size[1]
    return pt

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nheads, attn, dropout):
        super().__init__()
        assert d_model % nheads == 0
        self.d_k = d_model // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)
        self.attn = attn

    def forward(self, query, key, value, mask=None):
        """
        Transform the query, key, value into different heads, then apply the attention in parallel
        Args:
            query, key, value: size (N, S, C, T, D)
        Returns:
            (N, S, C, T, D)
        """
        nbatches = query.size(0)
        nspace = query.size(1)
        nchannel = query.size(2)
        # (N, h, S, C, d_k)
        query, key, value = \
            [l(x).view(x.size(0), x.size(1), x.size(2), self.nheads, self.d_k).permute(0, 3, 1, 2, 4)
             for l, x in zip(self.linears, (query, key, value))]

        # (N, h, S, C, d_k)
        x = self.attn(query, key, value, dropout=self.dropout)

        # (N, S, C, T, D)
        x = x.permute(0, 2, 3, 1, 4).contiguous() \
             .view(nbatches, nspace, nchannel, self.nheads * self.d_k)
        return self.linears[-1](x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

class SpaceAttentionBranch(nn.Module):
    def __init__(self, d_model=256, nheads=8, dim_feedforward=512, dropout=0.1, patch_size=[5,10],kernal=50,x_dim=40,emb_spatial_size=81,device='cuda:0'):
        super(SpaceAttentionBranch, self).__init__()
        self.patch_size = patch_size
        self.kernal = kernal
        self.x_dim =x_dim
        self.x_emb = x_embedding(x_dim, kernal, d_model, emb_spatial_size,device)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.space_attn = MultiHeadedAttention(d_model, nheads, SelfSpaceAttention, dropout)
        self.linear_output = nn.Linear(d_model, self.kernal)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def forward(self, x):
        N,C,H,W = x.shape
        x = unfold_StackOverChannel(x, self.patch_size)
        N, C, KK, HN, WH = x.shape
        x = x.reshape(N,C,KK, -1).permute(0, 3, 1, 2) # N,S,C,D
        x = self.x_emb(x)
        x = self.sublayer[0](x, lambda x: self.space_attn(x, x, x))
        x = self.sublayer[1](x, self.feed_forward)

        x = self.linear_output(x).permute(0, 2, 3, 1)  # (N, C, C_, S)

        # (N, T_tgt, C, C_, H_, W_)
        x = x.reshape(x.size(0), self.x_dim, self.kernal,
                                H // self.patch_size [0], W // self.patch_size [1])
        # (N, T_tgt, C, H, W)
        x = fold_tensor(x[:,None,:,:,:,:], output_size=(H, W), kernel_size=self.patch_size)

        return x.squeeze()

class ChannelAttentionBranch(nn.Module):
    def __init__(self, d_model=256, nheads=8, dim_feedforward=512, dropout=0.1, patch_size=[5,10],kernal=50,x_dim=40,emb_spatial_size=81,device='cuda:0'):
        super(ChannelAttentionBranch, self).__init__()
        self.patch_size = patch_size
        self.kernal = kernal
        self.x_dim =x_dim
        self.x_emb = x_embedding(x_dim, kernal, d_model, emb_spatial_size,device)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.channel_attn = MultiHeadedAttention(d_model, nheads, SelfChannelAttention, dropout)
        self.linear_output = nn.Linear(d_model, self.kernal)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def forward(self, x):
        N,C,H,W = x.shape
        x = unfold_StackOverChannel(x, self.patch_size)
        N, C, KK, HN, WH = x.shape
        x = x.reshape(N,C,KK, -1).permute(0, 3, 1, 2) # N,S,C,D
        x = self.x_emb(x)
        x = self.sublayer[0](x, lambda x: self.channel_attn(x, x, x))
        x = self.sublayer[1](x, self.feed_forward)

        x = self.linear_output(x).permute(0, 2, 3, 1)  # (N, C, C_, S)

        # (N, T_tgt, C, C_, H_, W_)
        x = x.reshape(x.size(0), self.x_dim, self.kernal,
                                H // self.patch_size [0], W // self.patch_size [1])
        # (N, T_tgt, C, H, W)
        x = fold_tensor(x[:,None,:,:,:,:], output_size=(H, W), kernel_size=self.patch_size)

        return x.squeeze()

class Model_Pre(nn.Module):

    def __init__(self, d_model=256, nheads=8, dim_feedforward=512, dropout=0.2, patch_size=[4,4],kernal=16,data_dim=3,emb_spatial_size=25,device='cuda:0'):
        super().__init__()

        self.x_emb = x_embedding(data_dim, kernal, d_model,
                                 emb_spatial_size, device)
        self.channel_attn = MultiHeadedAttention(d_model=d_model, nheads=nheads, attn=SelfChannelAttention, dropout=dropout)
        self.space_attn = MultiHeadedAttention(d_model=d_model, nheads=nheads, attn=SelfSpaceAttention, dropout=dropout)
        self.channel_space_attn = MultiHeadedAttention(d_model=d_model, nheads=nheads, attn=SelfChannelSpaceAttention, dropout=dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.linear_output = nn.Linear(d_model, kernal)
        self.patch_size = patch_size
        self.data_dim, self.kernal, self.patch_size = data_dim, kernal, patch_size

    def divided_channel_space_attn(self, query, key, value):
        # n = self.channel_attn(query, key, value)
        # return self.space_attn(n, n, n)
        return self.channel_space_attn(query, key, value)

    def forward(self, x):
        N, C, H, W = x.shape
        x = unfold_StackOverChannel(x, self.patch_size)
        N, C, KK, HN, WH = x.shape
        x = x.reshape(N, C, KK, -1).permute(0, 3, 1, 2)  # N,S,C,D
        x = self.x_emb(x)
        x = self.sublayer[0](x, lambda x: self.divided_channel_space_attn(x, x, x))
        x = self.sublayer[1](x, self.feed_forward)

        x = self.linear_output(x).permute(0, 2, 3, 1)  # (N, C, C_, S)

        # (N, C, C_, H_, W_)
        x = x.reshape(x.size(0), self.data_dim, self.kernal,
                      H // self.patch_size[0], W // self.patch_size[1])
        # (N, C, H, W)
        x = fold_tensor(x[:, None, :, :, :, :], output_size=(H, W), kernel_size=self.patch_size).squeeze()
        return x

class x_embedding(nn.Module):
    def __init__(self, input_dim, kernal, d_model, emb_spatial_size, max_len, device='cuda:0'):
        super().__init__()

        self.spatial_pos = torch.arange(emb_spatial_size)[None, :, None].to(device)
        self.emb_space = nn.Embedding(emb_spatial_size, d_model)

        self.channel_pos = torch.arange(input_dim)[None, None, :].to(device)
        self.emb_channel = nn.Embedding(input_dim, d_model)

        self.linear = nn.Linear(kernal, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Add temporal encoding and learnable spatial embedding to the input (after patch)
        Args:
            input x of size (N, S, C, C_)
        Returns:
            embedded input (N, S, C,  D)
        """
        assert len(x.size()) == 4
        embedded_space = self.emb_space(self.spatial_pos.to(configs.device)) # (1, S, 1, D)
        # print('embedded_space:',embedded_space[0,0,0,:2])
        embedded_channel = self.emb_channel(self.channel_pos.to(configs.device))  # (1, 1, C, D)
        # print('embedded_channel:', embedded_channel[0, 0, 0, :2])
        x = self.linear(x) + embedded_channel + embedded_space  # (N, S, C,  D)
        return self.norm(x)

# Attention Branch
# class AttentionBranch_1(nn.Module):
#
#     def __init__(self, nf, k_size=3):
#
#         super(AttentionBranch_1, self).__init__()
#         self.k1 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
#         self.sigmoid = nn.Sigmoid()
#         self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
#         self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
#
#     def forward(self, x):
#
#         y = self.k1(x)
#         y = self.lrelu(y)
#         y = self.k2(y)
#         y = self.sigmoid(y)
#
#
#         out = torch.mul(self.k3(x), y)
#         out = self.k4(out)
#
#         return out

class AttentionBranch_1(nn.Module):
    def __init__(self, nf, k_size=3):
        super(AttentionBranch_1, self).__init__()
        self.conv_3d = nn.Conv3d(1, 1, kernel_size=3, padding=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, c, h, w = x.size()
        x_reshape = x.reshape(n, 1, c, h, w)
        x_3d = self.sigmoid(self.conv_3d(x_reshape))
        x_squzzed = x_3d.reshape(n, c, h, w)
        return (self.scale * x_squzzed) * x + x

# class AttentionBranch_2(nn.Module):
#     def __init__(self, d_model=256, nheads=8, dim_feedforward=512, dropout=0.2, patch_size=[4,4],kernal=16,x_dim=40,emb_spatial_size=25,device='cuda:0'):
#         super(AttentionBranch_2, self).__init__()
#         self.patch_size = patch_size
#         self.kernal = kernal
#         self.x_dim =x_dim
#         self.x_emb = x_embedding(x_dim, kernal, d_model, emb_spatial_size,device)
#         self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
#         self.channel_attn = MultiHeadedAttention(d_model, nheads, SelfChannelAttention, dropout)
#         self.space_attn = MultiHeadedAttention(d_model, nheads, SelfSpaceAttention, dropout)
#         self.linear_output = nn.Linear(d_model, self.kernal)
#
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, dim_feedforward),
#             nn.ReLU(),
#             nn.Linear(dim_feedforward, d_model)
#             )
#
#     def divided_channel_space_attn(self, query, key, value):
#         n = self.channel_attn(query, key, value)
#         return self.space_attn(n, n, n)
#
#     def forward(self, x):
#         N,C,H,W = x.shape
#         x = unfold_StackOverChannel(x, self.patch_size)
#         N, C, KK, HN, WH = x.shape
#         x = x.reshape(N,C,KK, -1).permute(0, 3, 1, 2) # N,S,C,D
#         x = self.x_emb(x)
#         x = self.sublayer[0](x, lambda x: self.divided_channel_space_attn(x, x, x))
#         x = self.sublayer[1](x, self.feed_forward)
#
#         x = self.linear_output(x).permute(0, 2, 3, 1)  # (N, C, C_, S)
#
#         # (N, T_tgt, C, C_, H_, W_)
#         x = x.reshape(x.size(0), self.x_dim, self.kernal,
#                                 H // self.patch_size [0], W // self.patch_size [1])
#         # (N, T_tgt, C, H, W)
#         x = fold_tensor(x[:,None,:,:,:,:], output_size=(H, W), kernel_size=self.patch_size)
#
#         return x.squeeze()

class AttentionBranch_2(nn.Module):
    def __init__(self, d_model=256, nheads=8, dim_feedforward=512, dropout=0.2, patch_size=[4,4],kernal=16,x_dim=40,emb_spatial_size=25,device='cuda:0'):
        super(AttentionBranch_2, self).__init__()
        self.patch_size = patch_size
        self.kernal = kernal
        self.x_dim =x_dim
        self.x_emb = x_embedding(x_dim, kernal, d_model, emb_spatial_size,device)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 6)
        self.ChannelSpace_attn = MultiHeadedAttention(d_model, nheads, SelfChannelSpaceAttention, dropout)
        self.linear_output = nn.Linear(d_model, self.kernal)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
            )

    def divided_channel_space_attn(self, query, key, value):
        return self.ChannelSpace_attn(query, key, value)

    def forward(self, x):
        N,C,H,W = x.shape
        x = unfold_StackOverChannel(x, self.patch_size)
        N, C, KK, HN, WH = x.shape
        x = x.reshape(N,C,KK, -1).permute(0, 3, 1, 2) # N,S,C,D
        x = self.x_emb(x)
        x = self.sublayer[0](x, lambda x: self.divided_channel_space_attn(x, x, x))
        x = self.sublayer[1](x, self.feed_forward)
        x = self.sublayer[2](x, lambda x: self.divided_channel_space_attn(x, x, x))
        x = self.sublayer[3](x, self.feed_forward)
        x = self.sublayer[4](x, lambda x: self.divided_channel_space_attn(x, x, x))
        x = self.sublayer[5](x, self.feed_forward)

        x = self.linear_output(x).permute(0, 2, 3, 1)  # (N, C, C_, S)

        # (N, T_tgt, C, C_, H_, W_)
        x = x.reshape(x.size(0), self.x_dim, self.kernal,
                                H // self.patch_size [0], W // self.patch_size [1])
        # (N, T_tgt, C, H, W)
        x = fold_tensor(x[:,None,:,:,:,:], output_size=(H, W), kernel_size=self.patch_size)

        return x.squeeze()

class AttentionBranch_3(nn.Module):

    def __init__(self, nf, k_size=3):

        super(AttentionBranch_3, self).__init__()
        self.CA = ChannelAttention(in_planes = nf,ratio = 20)
        self.SA = SpatialAttention(k_size)

    def forward(self, x):
        a = self.CA(x)
        y = x * a
        b = self.SA(x)
        y = y * b

        return y

class AAB(nn.Module):

    def __init__(self, nf, reduction=4, K=3, t=30):
        super(AAB, self).__init__()
        self.t=t
        self.K = K

        self.conv_first = nn.Conv2d(nf, nf, kernel_size=1, bias=False)
        self.conv_last = nn.Conv2d(nf, nf, kernel_size=1, bias=False)        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Attention Dropout Module
        self.ADM = nn.Sequential(
            nn.Linear(nf, nf // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nf // reduction, self.K, bias=False),
        )
        
        # attention branch
        self.attention_1 = AttentionBranch_1(nf)
        self.attention_2 = AttentionBranch_2(d_model=256, nheads=8, dim_feedforward=512, dropout=0.2, patch_size=[4,4],kernal=16,x_dim=40,emb_spatial_size=25,device='cuda:0')
        self.attention_3 = AttentionBranch_3(nf)
        
        # non-attention branch
        # 3x3 conv for A2N
        self.non_attention = nn.Conv2d(nf, nf, kernel_size=3, padding=(3 - 1) // 2, bias=False)         
        # 1x1 conv for A2N-M (Recommended, fewer parameters)
        # self.non_attention = nn.Conv2d(nf, nf, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        a, b, c, d = x.shape

        x = self.conv_first(x)
        x = self.lrelu(x)

        # Attention Dropout
        y = self.avg_pool(x).view(a,b)
        y = self.ADM(y)
        ax = F.softmax(y/self.t, dim = 1)

        # attention_1 = self.attention_1(x)
        non_attention = self.non_attention(x)
        attention_2 = self.attention_2(x)
        attention_3 = self.attention_3(x)

        # print(attention_3.shape)
        # plt.figure(figsize=(4, 4))
        # plt.subplot(121)
        # plt.title('CBAM', fontdict={'size': 10.5})
        # plt.imshow(attention_2[0].mean(0).cpu(), cmap='RdBu')
        # # plt.clim(0, 1)
        # plt.axis('off')
        # plt.subplot(122)
        # plt.title('SA', fontdict={'size': 10.5})
        # plt.imshow(attention_3[0].mean(0).cpu(), cmap='RdBu')
        # # plt.clim(0, 1)
        # plt.axis('off')
        # plt.show()
    
        x = attention_3 * ax[:,0].view(a,1,1,1) + non_attention * ax[:,1].view(a,1,1,1) + attention_2 * ax[:,2].view(a,1,1,1)
        x = self.lrelu(x)

        # print('[', np.array(ax[:, 1].mean().detach().cpu()), ',', np.array(ax[:, 0].mean().detach().cpu()), ',',
        #       np.array(ax[:, 2].mean().detach().cpu()), ']', ',')

        out = self.conv_last(x)
        out += residual

        return out


class Layer_embedding(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.layer_pos = torch.arange(configs.blocknum)[None, None,:].to(configs.device)
        self.emb_layer = nn.Embedding(configs.blocknum, configs.d_model)

        self.linear = nn.Linear(configs.d_pix, configs.d_model)
        self.norm = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        """
        Add temporal encoding and learnable spatial embedding to the input (after patch)
        Args:
            input x of size (N,C,L,D)
        Returns:
            embedded input (N,C,L,D)
        """
        assert len(x.size()) == 4
        embedded_layer = self.emb_layer(self.layer_pos) # (1, 1, L, D)
        x = self.linear(x) + embedded_layer
        return self.norm(x)

def SelfLayerAttention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)

class AttentionBranch_Layer(nn.Module):
    def __init__(self, configs):
        super(AttentionBranch_Layer, self).__init__()

        self.layer_emb = Layer_embedding(configs)
        self.sublayer = clones(SublayerConnection(configs.d_model, configs.dropout), 6)
        self.layer_attn = MultiHeadedAttention(configs.d_model, configs.nheads, SelfLayerAttention, configs.dropout)
        self.linear_output = nn.Linear(configs.d_model, configs.d_pix)

        self.feed_forward = nn.Sequential(
            nn.Linear(configs.d_model, configs.dim_feedforward),
            nn.ReLU(),
            nn.Linear(configs.dim_feedforward, configs.d_model)
            )

    def forward(self, x):
        L,N,C,H,W = x.shape
        x = x.reshape(L,N,C, -1).permute(1,2,0,3) # N,C,L,D
        x = self.layer_emb(x)
        x = self.sublayer[0](x, lambda x: self.layer_attn(x, x, x))
        x = self.sublayer[1](x, self.feed_forward)
        x = self.sublayer[2](x, lambda x: self.layer_attn(x, x, x))
        x = self.sublayer[3](x, self.feed_forward)
        x = self.sublayer[4](x, lambda x: self.layer_attn(x, x, x))
        x = self.sublayer[5](x, self.feed_forward)

        x = self.linear_output(x).permute(2,0,1,3).reshape(L,N,C,H,W)

        return x


class SDSRNET(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        
        self.configs = configs
        
        in_nc  = 3
        out_nc = 3
        nf = 40
        unf = 24
        nb = self.configs.blocknum

        # AAB
        AAB_block_f = functools.partial(AAB, nf=nf)
        self.scale = self.configs.scale
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.model_first = Model_Pre()
        
        ### main blocks
        self.AAB_trunk = make_layer(AAB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ### layer Attention blocks
        layer_att_f = functools.partial(AttentionBranch_Layer, configs=self.configs)
        self.layer_att = make_layer(layer_att_f,5)

        #### upsampling
        self.upconv1 = nn.Conv2d(nf*nb, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)

        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float))
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        elif self.scale == 8:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.upconv3 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att3 = PA(unf)
            self.HRconv3 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.la_conv = nn.Conv2d(nf*nb, nf, kernel_size=3, padding=1, bias=False)

    def forward(self, x):

        # fea = self.model_first(x)
        # fea = self.conv_first(fea)

        fea = self.conv_first(x)

        y = fea
        ml_r = []
        for i in range(len(self.AAB_trunk)):
            y = self.AAB_trunk[i](y)
            ml_r.append(y)

        ml_att = torch.tensor([item.cpu().detach().numpy() for item in ml_r]).to(self.configs.device)

        ml_att = self.layer_att(ml_att).transpose(1,0).flatten(start_dim=1, end_dim=2)


        trunk = self.trunk_conv(y)

        # trunk = self.trunk_conv(self.AAB_trunk(fea))

        # fea = fea + trunk + self.la_conv(ml_att)
        fea = fea.repeat(1,self.configs.blocknum,1,1) + trunk.repeat(1,self.configs.blocknum,1,1) + ml_att
        
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
        elif self.scale == 8:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
            fea = self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att3(fea))
            fea = self.lrelu(self.HRconv3(fea))
        
        out = self.conv_last(fea)
        
        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR

        return out

    def get_last_shared_layer(self):
        return self.conv_last