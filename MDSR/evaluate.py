import torch.nn as nn
import pytorch_ssim
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from math import exp

def image_lr_normalization_maxmin(srimage,hrimage):
    N, C, H, W = hrimage.shape
    image = torch.cat((torch.tensor(srimage).flatten(start_dim=2,end_dim=3),torch.tensor(hrimage).flatten(start_dim=2,end_dim=3)),-1)

    for i in range(N):
        cmax_list, cmin_list = [], []
        for j in range(C):
            simage = np.array(image[i,j])
            s_max,s_min = max(simage), min(simage)
            cmax_list.append(s_max)
            cmin_list.append(s_min)
        cmax, cmin = np.array(cmax_list)[:,None], np.array(cmin_list)[:,None]
        cnin = np.concatenate([cmax, cmin], 1)[None]
        if(i == 0):
            max_min = cnin
        else:
            max_min = np.concatenate([max_min,cnin],0) # N,C,2
    max_adj, min_adj = np.zeros((N, C, H, W)), np.zeros((N, C, H, W))
    for i in range(N):
        for j in range(C):
            max_adj[i,j] = np.full((1, 1, H, W),max_min[i,j,0])
            min_adj[i, j] = np.full((1, 1, H, W), max_min[i, j, 1])
    srimage = (srimage-min_adj)/(max_adj-min_adj)
    hrimage = (hrimage - min_adj) / (max_adj - min_adj)
    return np.array(srimage),np.array(hrimage)

L1 = nn.L1Loss()

L2 = nn.MSELoss()

def PSNR(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return torch.tensor(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(img1, img2, window_size=11, size_average=True):
    img1 = img1 * 0.5 + 0.5
    img2 = img2 * 0.5 + 0.5

    if len(img1.size()) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.size()) == 3:
        img2 = img2.unsqueeze(0)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

data = np.load('result.npz')
ilr,sr,hr = torch.tensor(data['ilr']),torch.tensor(data['sr']),torch.tensor(data['hr'])

image_sr,image_hr = image_lr_normalization_maxmin(np.array(sr),np.array(hr))
image_sr,image_hr = torch.tensor(image_sr)*255,torch.tensor(image_hr)*255

print('MAE:',L1(sr[:,:1],hr[:,:1]).item(),L1(sr[:,1:2],hr[:,1:2]).item(),L1(sr[:,2:3],hr[:,2:3]).item(),L1(sr,hr).item())
print('MSE:',L2(sr[:,:1],hr[:,:1]).item(),L2(sr[:,1:2],hr[:,1:2]).item(),L2(sr[:,2:3],hr[:,2:3]).item(),L2(sr,hr).item())
print('PSNR:',PSNR(image_sr[:,:1],image_hr[:,:1]).item(),PSNR(image_sr[:,1:2],image_hr[:,1:2]).item(),PSNR(image_sr[:,2:3],image_hr[:,2:3]).item(),PSNR(image_sr,image_hr).item())
print('SSIM:',SSIM(image_sr[:,:1],image_hr[:,:1]).item(),SSIM(image_sr[:,1:2],image_hr[:,1:2]).item(),SSIM(image_sr[:,2:3],image_hr[:,2:3]).item(),SSIM(image_sr,image_hr).item())