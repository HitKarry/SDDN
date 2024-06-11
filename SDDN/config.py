import torch

class Configs:
    def __init__(self):
        pass

configs = Configs()

# trainer related
configs.device = torch.device('cuda:1')
configs.batch_size = 3
configs.batch_size_test = 3
configs.lr = 0.0005
configs.weight_decay = 0
configs.display_interval = 60
configs.num_epochs = 1000
configs.early_stopping = True
configs.patience = 10
configs.gradient_clipping = False
configs.clipping_threshold = 1.

# data related
configs.d_pix = 20*20

# model
configs.scale = 4
configs.blocknum = 10
configs.d_model = 256
configs.dropout = 0.2
configs.nheads = 4
configs.dim_feedforward = 512

# grad norm
configs.tau = 0.1
configs.eps = 1e-12