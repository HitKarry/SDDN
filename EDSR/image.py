import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
from collections import OrderedDict
from skimage import io, data, color, util
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

data = np.load('result.npz')
data_lr = np.load('../data/lr_data_rh850_z500_t850_X8.npz')
lr = data_lr['data'][:3]
ilr,sr,hr = data['ilr'][:3],data['sr'][:3],data['hr'][:3] # (746, 3, 360, 720)
print(lr.shape,ilr.shape,sr.shape,hr.shape)


# def image_normalization_maxmin(image):
#     N, C, H, W = image.shape
#     for i in range(N):
#         cmax_list, cmin_list = [], []
#         for j in range(C):
#             simage = np.array(image[i,j]).flatten()
#             s_max,s_min = max(simage), min(simage)
#             cmax_list.append(s_max)
#             cmin_list.append(s_min)
#         cmax, cmin = np.array(cmax_list)[:,None], np.array(cmin_list)[:,None]
#         cnin = np.concatenate([cmax, cmin], 1)[None]
#         if(i == 0):
#             max_min = cnin
#         else:
#             max_min = np.concatenate([max_min,cnin],0) # N,C,2
#     max_adj, min_adj = max_min[:,:,0,None,None], max_min[:,:,1,None,None]
#     image = (image-min_adj)/(max_adj-min_adj)
#     return np.array(image)

def image_lr_normalization_maxmin(image):
    N, C, H, W = image.shape
    for i in range(N):
        cmax_list, cmin_list = [], []
        for j in range(C):
            simage = np.array(image[i,j]).flatten()
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
    image = (image-min_adj)/(max_adj-min_adj)
    return np.array(image)

i,j,k = 0,1,2
image_lr = image_lr_normalization_maxmin(np.array(lr)).transpose(0,2,3,1)
image_ilr = image_lr_normalization_maxmin(np.array(ilr)).transpose(0,2,3,1)
image_sr = image_lr_normalization_maxmin(np.array(sr)).transpose(0,2,3,1)
image_hr = image_lr_normalization_maxmin(np.array(hr)).transpose(0,2,3,1)
image_lr = np.concatenate([image_lr[:,:,:,i][:,:,:,None],image_lr[:,:,:,j][:,:,:,None],image_lr[:,:,:,k][:,:,:,None]],-1)
image_ilr = np.concatenate([image_ilr[:,:,:,i][:,:,:,None],image_ilr[:,:,:,j][:,:,:,None],image_ilr[:,:,:,k][:,:,:,None]],-1)
image_sr = np.concatenate([image_sr[:,:,:,i][:,:,:,None],image_sr[:,:,:,j][:,:,:,None],image_sr[:,:,:,k][:,:,:,None]],-1)
image_hr = np.concatenate([image_hr[:,:,:,i][:,:,:,None],image_hr[:,:,:,j][:,:,:,None],image_hr[:,:,:,k][:,:,:,None]],-1)

for i in range(len(image_hr)):
    io.imsave('../images/lr_image_' + str(i) + '.jpg', util.img_as_ubyte(image_lr[i]))
    io.imsave('../images/ilr_image_' + str(i) + '.jpg', util.img_as_ubyte(image_ilr[i]))
    io.imsave('../images/hr_image_' + str(i) + '.jpg', util.img_as_ubyte(image_hr[i]))
    io.imsave('../images/sr_image_' + str(i) + '.jpg', util.img_as_ubyte(image_sr[i]))