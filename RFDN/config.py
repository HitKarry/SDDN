import torch

class Configs:
    def __init__(self):
        pass

configs = Configs()

configs.device = torch.device('cuda:1')

# RCAN
configs.in_nc = 3
configs.nf = 50
configs.num_modules = 4
configs.out_nc = 3
configs.scale = 4

configs.lr = 5e-4
configs.lr_decay = 200
configs.decay_type ='step'
configs.gamma = 0.5
configs.optimizer = 'ADAM'
configs.momentum = 0.9
configs.beta1 = 0.9
configs.beta2 = 0.999
configs.epsilon = 1e-8
configs.weight_decay = 0
configs.reduction = 16

configs.batch_size = 5
configs.batch_size_test = 5
configs.display_interval = 60
configs.num_epochs = 1000
configs.early_stopping = True
configs.patience = 10
configs.gradient_clipping = False
configs.clipping_threshold = 1.









# # trainer related
# configs.device = torch.device('cuda:0')
# configs.batch_size = 5
# configs.batch_size_test = 5
# configs.lr = 0.001
# configs.weight_decay = 0
# configs.display_interval = 60
# configs.num_epochs = 100
# configs.early_stopping = True
# configs.patience = 10
# configs.gradient_clipping = False
# configs.clipping_threshold = 1.
#
# # data related
#
#
# # model
# configs.scale = 8
# configs.blocknum = 20
#











# # trainer related
# configs.n_cpu = 0
# configs.device = torch.device('cuda:0')
# configs.batch_size_test = 100
# configs.batch_size = 8
# #configs.lr = 0.001
# configs.weight_decay = 0
# configs.display_interval = 120
# configs.num_epochs = 100
# configs.early_stopping = True
# configs.patience = 3
# configs.gradient_clipping = False
# configs.clipping_threshold = 1.

# # lr warmup
# configs.warmup = 3000

# # data related
# configs.input_dim = 1 # 4
# configs.output_dim = 1

# configs.input_length = 12
# configs.output_length = 26

# configs.input_gap = 1
# configs.pred_shift = 24

# # model
# configs.scale = 4
# configs.d_model = 256
# configs.patch_size = (2, 3)
# configs.emb_spatial_size = 12*16
# configs.nheads = 4
# configs.dim_feedforward = 512
# configs.dropout = 0.2
# configs.num_encoder_layers = 3
# configs.num_decoder_layers = 3
# configs.ssr_decay_rate = 3.e-4