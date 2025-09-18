import torch
import numpy as np
from torch.utils.data import Dataset
from han import HAN
from config import configs
import math
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import datetime
import utility

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = HAN(configs).to(configs.device)
net = torch.load('checkpoint.chk')
model.load_state_dict(net['net'])
model.eval()

data = DataLoader(dataset_eval, batch_size=3, shuffle=False)

starttime = datetime.datetime.now()
with torch.no_grad():
    for i,(lr,hr) in enumerate(data):
        pred_temp = model(lr.float().to(configs.device))
        if i == 0:
            pred = pred_temp
            label = hr
        else:
            pred = torch.cat((pred, pred_temp), 0)
            label = torch.cat((label, hr), 0)
endtime = datetime.datetime.now()
print('SPEND TIME:', endtime - starttime)


np.savez('result.npz', sr=pred.cpu()*data_std[None, :, None, None]+data_mean[None, :, None, None], hr=label.cpu()*data_std[None, :, None, None]+data_mean[None, :, None, None])