import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import pickle
import torch
from collections import OrderedDict
# from skimage import io, data, color, util
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

DATADIR = './data/'

z500 = xr.open_mfdataset(f'{DATADIR}geopotential_500/*.nc', combine='by_coords')
rh850 = xr.open_mfdataset(f'{DATADIR}relative_humidity_850/*.nc', combine='by_coords')
t850 = xr.open_mfdataset(f'{DATADIR}temperature_850/*.nc', combine='by_coords')

datasets = [rh850, z500, t850]
data_hr = np.array(xr.merge(datasets,compat='override').sel(time=slice("2017","2021")).to_array().transpose('time','variable', 'latitude', 'longitude')) # (N,C,H,W)

data_hr = torch.tensor(data_hr[:,:,400:560,380:540])

AvgPool = nn.AvgPool2d((2,2), stride=2)
data_lr_X2 = AvgPool(data_hr)
data_lr_X4 = AvgPool(AvgPool(data_hr))
data_lr_X8 = AvgPool(AvgPool(AvgPool(data_hr)))

np.savez('hr_data_rh850_z500_t850.npz',data=data_hr)
np.savez('lr_data_rh850_z500_t850_X2.npz',data=data_lr_X2)
np.savez('lr_data_rh850_z500_t850_X4.npz',data=data_lr_X4)
np.savez('lr_data_rh850_z500_t850_X8.npz',data=data_lr_X8)


