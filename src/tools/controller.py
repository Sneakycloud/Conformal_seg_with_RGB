
import logging
import numpy as np
import torch
import train
from torchvision import transforms

import utils.data_handling as dh
import utils.unet as unet
import utils.util as util

import h5py

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#datapath = "/home/beax22tr/nyu_depth_v2_labeled.mat"
datapath = "/home/beax22tr/SUN_RGBD"
#datapath = "../../../../home/data/epsilon/nyu_depth_v2_labeled.mat"

epochs = 50
batch_size = 16
#nyu_classes = list(range(255))

#gets number of classes
print("TMC training started")
train.rgbd_train(datapath, "./logs", 42, epochs, batch_size, class_list=None, mode="TMC")
print("RGBD training started")
train.rgbd_train(datapath, "./logs", 42, epochs, batch_size, class_list=None, mode="RGBD")
print("D training started")
train.rgbd_train(datapath, "./logs", 42, epochs, batch_size, class_list=None, mode="D")
print("RGB training started")
train.rgbd_train(datapath, "./logs", 42, epochs, batch_size, class_list=None, mode="RGB")
print("All training completed")
