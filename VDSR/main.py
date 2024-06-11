import torch
import numpy as np
from torch.utils.data import Dataset
from vdsr import VDSR
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


class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        torch.manual_seed(5)
        self.network = VDSR(configs).to(configs.device)
        self.optimizer = utility.make_optimizer(configs, self.network)
        self.scheduler = utility.make_scheduler(configs, self.optimizer)
        self.l1loss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.z, self.t, self.w = 'z', 't', 'w'

    # def loss(self, y_pred, y_true, idx):
    #     if idx == 'z':
    #         idx = 0
    #     if idx == 't':
    #         idx = 1
    #     if idx == 'w':
    #         idx = 2
    #     mse = self.l1loss(y_pred[:, idx],y_true[:, idx])
    #     return mse

    def loss(self, y_pred, y_true, idx):
        mse = self.mseloss(y_pred, y_true)
        return mse

    def evaloss(self, y_pred, y_true, idx):
        if idx == 'z':
            idx = 0
        if idx == 't':
            idx = 1
        if idx == 'w':
            idx = 2
        mse = self.mseloss(y_pred[:, idx],y_true[:, idx])
        return mse

    # def train_once(self, lr, hr):
    #     sr = self.network(lr.float().to(self.device))
    #     self.optimizer.zero_grad()
    #     loss_z = self.loss(sr, hr.float().to(self.device), self.z)
    #     loss_t = self.loss(sr, hr.float().to(self.device), self.t)
    #     loss_w = self.loss(sr, hr.float().to(self.device), self.w)
    #
    #     loss = loss_z + loss_t + loss_w
    #     loss.backward()
    #     if configs.gradient_clipping:
    #         nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
    #     self.optimizer.step()
    #     return loss.item(), sr

    def train_once(self, lr, hr):
        sr = self.network(lr.float().to(self.device))
        self.optimizer.zero_grad()
        loss = self.loss(sr, hr.float().to(self.device), self.z)

        loss.backward()
        if configs.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), configs.clipping_threshold)
        self.optimizer.step()
        return loss.item(), sr

    def test(self, dataloader_test):
        sr_pred = []
        f_loss1,f_loss2,s = 0,0,0
        self.network.eval()
        with torch.no_grad():
            for i,(lr,hr) in enumerate(dataloader_test):
                sr = self.network(lr.float().to(self.device))
                sr_inter = F.interpolate(lr.float().to(self.device), scale_factor=self.configs.scale, mode='bilinear', align_corners=False)
                # data_mean = torch.tensor([56.263893, 56217.297, 283.6241]).to(self.device)
                # data_std = torch.tensor([26.381908, 1735.2358, 8.533371]).to(self.device)
                # sr = sr * data_std[None, :, None, None] + data_mean[None, :, None, None]
                # sr_inter = sr_inter * data_std[None, :, None, None] + data_mean[None, :, None, None]
                # hr = hr.to(self.device) * data_std[None, :, None, None] + data_mean[None, :, None, None]
                loss1_z = self.evaloss(sr, hr.float().to(self.device), self.z)
                loss1_t = self.evaloss(sr, hr.float().to(self.device), self.t)
                loss1_w = self.evaloss(sr, hr.float().to(self.device), self.w)
                loss2_z = self.evaloss(sr_inter, hr.float().to(self.device), self.z)
                loss2_t = self.evaloss(sr_inter, hr.float().to(self.device), self.t)
                loss2_w = self.evaloss(sr_inter, hr.float().to(self.device), self.w)

                loss1 = loss1_z + loss1_w + loss1_t
                loss2 = loss2_z + loss2_w + loss2_t
                f_loss1 = f_loss1 + loss1
                f_loss2 = f_loss2 + loss2
                s += 1
                sr_pred.append(sr)

        return torch.cat(sr_pred, dim=0),f_loss1/s,f_loss2/s

    def train(self, dataset_train, dataset_eval, chk_path):
        torch.manual_seed(0)
        print('loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        print('loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)

        count = 0
        best = math.inf
        for i in range(self.configs.num_epochs):
            print('\nepoch: {0}'.format(i+1))
            # train
            self.network.train()
            for j,(lr,hr) in enumerate(dataloader_train):
                batch_loss,sr = self.train_once(lr,hr)

                if j % self.configs.display_interval == 0:
                    print('batch training loss: {:.3f}'.format(batch_loss))

                # increase the number of evaluations in order not to miss the optimal point
                if (i+1 >= 11) and (j+1)%120 == 0:
                    _, sc_loss1, _ = self.test(dataloader_eval)
                    self.network.train()
                    print('epoch eval loss: {:.4f}'.format(sc_loss1))
                    if sc_loss1 < best:
                        self.save_model('checkpoint_' + str(np.array(sc_loss1.cpu())) + chk_path)
                        best = sc_loss1
                        count = 0

            self.scheduler.step()

            # evaluation
            _, loss_eval, loss_eval2 = self.test(dataloader_eval)
            print('epoch eval loss: {:.3f}, inter loss: {:.3f}'.format(loss_eval,loss_eval2))
            if loss_eval >= best:
                count += 1
                print('eval loss is not reduceed for {} epoch'.format(count))
            else:
                count = 0
                print('eval loss is reduceed from {:.5f} to {:.5f}, saving model'.format(best, loss_eval))
                self.save_model('checkpoint_' + str(np.array(loss_eval.cpu())) + chk_path)
                best = loss_eval

            if count == self.configs.patience:
                print('early stopping reached, minimum loss is {:5f}'.format(best))
                break


    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)

    def save_model(self, path):
        torch.save({'net': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)

class dataset_generator(Dataset):
    def __init__(self, data_lr, data_hr):
        super().__init__()

        self.data_lr = data_lr
        self.data_hr = data_hr
        assert self.data_lr.shape[0] == self.data_hr.shape[0]

    def GetDataShape(self):
        return {'data_lr': self.data_lr.shape,
                'data_hr': self.data_hr.shape}

    def __len__(self,):
        return self.data_lr.shape[0]

    def __getitem__(self, idx):
        return self.data_lr[idx], self.data_hr[idx]

def data_normalization_zscore(lr,hr):
    lr,hr = np.array(lr),np.array(hr)
    lr_mean, lr_std = lr.mean((0, 2, 3)), lr.std((0, 2, 3))
    lr = (lr - lr_mean[None,:,None,None]) / lr_std[None,:,None,None]
    hr = (hr - lr_mean[None, :, None, None]) / lr_std[None, :, None, None]
    return lr, hr, lr_mean, lr_std

if __name__ == '__main__':
    print(configs.__dict__)

    print('\nreading data')
    lr = np.load('../../data/lr_data_rh850_z500_t850_X8.npz')
    hr = np.load('../../data/lr_data_rh850_z500_t850_X2.npz')
    data_lr,data_hr = lr['data'], hr['data']

    print('\ndata normalization')
    data_lr, data_hr, data_mean, data_std = data_normalization_zscore(data_lr,data_hr)
    print('data_mean',data_mean)
    print('data_std', data_std)

    # _, _, data_mean, data_std = data_normalization_zscore(data_lr, data_hr)
    # print('data_mean', data_mean)
    # print('data_std', data_std)

    print('processing training set')
    dataset_train = dataset_generator(data_lr[:1080],data_hr[:1080])
    print(dataset_train.GetDataShape())

    print('processing eval set')
    dataset_eval = dataset_generator(data_lr[1080:],data_hr[1080:])
    print(dataset_eval.GetDataShape())

    del data_lr, data_hr

    # trainer = Trainer(configs)
    # trainer.save_configs('config_train.pkl')
    # trainer.train(dataset_train, dataset_eval, '.chk')

    #########################################################################################################################################

    model = VDSR(configs).to(configs.device)
    net = torch.load('checkpoint_0.050941378.chk')
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

    # ILR = F.interpolate(torch.tensor(data_lr), scale_factor=8, mode='bilinear', align_corners=False)

    np.savez('result.npz', sr=pred.cpu()*data_std[None, :, None, None]+data_mean[None, :, None, None], hr=label.cpu()*data_std[None, :, None, None]+data_mean[None, :, None, None])
