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

data = np.load('./data.npz')

data_hr = data['data'][:,:,:-2,:-2]

data_hr = torch.tensor(np.nan_to_num(data_hr))


AvgPool = nn.AvgPool2d((2,2), stride=2)
data_lr_X2 = AvgPool(data_hr)
data_lr_X4 = AvgPool(AvgPool(data_hr))
data_lr_X8 = AvgPool(AvgPool(AvgPool(data_hr)))


np.savez('hr_cldas_data_tmp_prs.npz',data=data_hr)
np.savez('lr_cldas_data_tmp_prs_X2.npz',data=data_lr_X2)
np.savez('lr_cldas_data_tmp_prs_X4.npz',data=data_lr_X4)
np.savez('lr_cldas_data_tmp_prs_X8.npz',data=data_lr_X8)

