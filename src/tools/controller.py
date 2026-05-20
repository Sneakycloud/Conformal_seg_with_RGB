
import logging
import numpy as np
import torch
import train
from torchvision import transforms

import utils.data_handling as dh
import utils.unet as unet
import utils.util as util

import h5py

datapath = "../../data/nyu_v2/nyu_depth_v2_labeled.mat"
#datapath = "../../../../home/data/epsilon/nyu_depth_v2_labeled.mat"

#gets number of classes
print("RGB training started")
train.rgbd_train("../../data/nyu_v2/nyu_depth_v2_labeled.mat", "../../logs", 42, 2, 64, class_list=torch.tensor([x for x in range(256)], mode="RGB"))
print("D training started")
train.rgbd_train("../../data/nyu_v2/nyu_depth_v2_labeled.mat", "../../logs", 42, 2, 64, class_list=torch.tensor([x for x in range(256)], mode="D"))
print("RGBD training started")
train.rgbd_train("../../data/nyu_v2/nyu_depth_v2_labeled.mat", "../../logs", 42, 2, 64, class_list=torch.tensor([x for x in range(256)], mode="RGBD"))
print("TMC training started")
train.rgbd_train("../../data/nyu_v2/nyu_depth_v2_labeled.mat", "../../logs", 42, 2, 64, class_list=torch.tensor([x for x in range(256)], mode="TMC"))
print("All training completed")