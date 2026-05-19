
import logging
import numpy as np
import torch
import train
from torchvision import transforms

import utils.data_handling as dh
import utils.unet as unet
import utils.util as util

import h5py


#gets number of classes
train.rgbd_train("../../data/nyu_v2/nyu_depth_v2_labeled.mat", "../../logs", 42, 2, 64, torch.tensor([x for x in range(256)]))