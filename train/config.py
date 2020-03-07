import numpy as np
import os
from easydict import EasyDict as edict

config = edict()

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = False
config.net_se = 0
config.net_act = 'relu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1,4,6,2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = False
config.data_cutoff = False
config.data_color = 0
config.data_images_filter = 0
config.count_flops = True
config.memonger = False #not work now


# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.net_act = 'prelu'
network.r100.num_layers = 100

network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 512
network.y1.net_act = 'relu'
network.y1.net_output = 'GDC'

# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = '/mnt/ssd2/Datasets/faces_emore' #'../datasets/faces_emore'
dataset.emore.num_classes = 85742
dataset.emore.image_shape = (112,112,3)
dataset.emore.val_targets = ['lfw', 'cfp_fp', 'agedb_30']


dataset.emore_soft = edict()
dataset.emore_soft.dataset = 'emore_soft'
dataset.emore_soft.dataset_path = '/mnt/ssd2/Datasets/faces_emore_bin'
dataset.emore_soft.num_classes = 85742
dataset.emore_soft.image_shape = (112,112,3)
dataset.emore_soft.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

loss = edict()

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.triplet_distillation_L2 = edict()
loss.triplet_distillation_L2.loss_name = 'triplet_distillation_L2'
loss.triplet_distillation_L2.triplet_alpha = 0.35
loss.triplet_distillation_L2.images_per_identity = 18
loss.triplet_distillation_L2.triplet_bag_size = 7200
loss.triplet_distillation_L2.triplet_max_ap = 0.0
loss.triplet_distillation_L2.per_batch_size = 180
loss.triplet_distillation_L2.lr = 0.001
loss.triplet_distillation_L2.Mmax = 0.5
loss.triplet_distillation_L2.Mmin = 0.2

loss.triplet_distillation_cos = edict()
loss.triplet_distillation_cos.loss_name = 'triplet_distillation_cos'
loss.triplet_distillation_cos.triplet_alpha = 0.35
loss.triplet_distillation_cos.images_per_identity = 18
loss.triplet_distillation_cos.triplet_bag_size = 7200
loss.triplet_distillation_cos.triplet_max_ap = 0.0
loss.triplet_distillation_cos.per_batch_size = 180
loss.triplet_distillation_cos.lr = 0.001
loss.triplet_distillation_cos.Mmax = 0.5
loss.triplet_distillation_cos.Mmin = 0.2

loss.margin_distillation = edict()
loss.margin_distillation.loss_name = 'margin_distillation'
loss.margin_distillation.loss_s = 64.0
loss.margin_distillation.Mmax = 0.5
loss.margin_distillation.Mmin = 0.2
loss.margin_distillation.teacher_model = './models/r100-arcface-emore/model'

loss.margin_base_with_T = edict()
loss.margin_base_with_T.loss_name = 'margin_base_with_T'
loss.margin_base_with_T.loss_s = 64.0
loss.margin_base_with_T.loss_m1 = 1.0
loss.margin_base_with_T.loss_m2 = 0.5
loss.margin_base_with_T.loss_m3 = 0.0
loss.margin_base_with_T.T = 4
loss.margin_base_with_T.alpha = 0.95
loss.margin_base_with_T.teacher_model = './models/r100-arcface-emore/model'

loss.angular_distillation = edict()
loss.angular_distillation.loss_name = 'angular_distillation'
loss.angular_distillation.loss_s = 64.0
loss.angular_distillation.alpha = 1.0

# default settings
default = edict()

# default network
default.network = 'r100'
default.pretrained = ''
default.pretrained_epoch = 1
# default dataset
default.dataset = 'emore'
default.loss = 'arcface'
default.frequent = 20
default.verbose = 4000
default.kvstore = 'device'

default.end_epoch = 10000
default.lr = 0.1
default.wd = 0.0005
default.mom = 0.9
default.per_batch_size = 256
default.T = 4
default.ckpt = 3
default.lr_steps = '100000,160000,220000'
default.models_root = './models'


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
      config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

