import logging
import os

import numpy as np
import torch
import torchvision
import h5py
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from os.path import dirname, join as pjoin

class RGBD_Segmentation_Dataset(Dataset):
    def __init__(self, folder_path, transform):
        super(RGBD_Segmentation_Dataset, self).__init__()
        with h5py.File(folder_path,'r') as f:
            self.rgb_images = torch.from_numpy(np.array(f["images"],dtype=np.uint8)).float() # N x C x H x W
            self.depth_images = torch.from_numpy(np.expand_dims(np.array(f["depths"],dtype=np.uint8),axis=1)).float() #The expand dims is to get N x C x H x W
            self.mask_images = torch.from_numpy(np.expand_dims(np.array(f["labels"],dtype=np.uint8),axis=1)).float() #The expand dims is to get N x C x H x W
        self.transform = transform
        self.class_count()
        
    def __getitem__(self, index):
            rgb = self.rgb_images[index]
            depth = self.depth_images[index]
            mask = self.mask_images[index]
            
            if self.transform:
                rgb, depth, mask = self.transform((rgb, depth, mask))    
            
            return rgb, depth, mask

    def __len__(self):
        return self.rgb_images.shape[0]
    
    def class_count(self):
        self.classes = torch.unique(self.mask_images).numpy()
        return self.classes
    

#https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.probaility = p
        
    def __call__(self, sample):
        rgb, depth, mask = sample
        
        if torch.randint(100,size=(1,)) < self.probaility*100:
            rgb = transforms.functional.vflip(rgb)
            depth = transforms.functional.vflip(depth)
            mask = transforms.functional.vflip(mask)
            
        return (rgb,depth,mask)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.probaility = p

    def __call__(self, sample):
        rgb, depth, mask = sample
        
        if torch.randint(100,size=(1,)) < self.probaility*100:
            rgb = transforms.functional.hflip(rgb)
            depth = transforms.functional.hflip(depth)
            mask = transforms.functional.hflip(mask)
            
        return (rgb,depth,mask)

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        rgb, depth, mask = sample

        h, w = rgb.shape[1:]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h + 1, (1,))
        left = torch.randint(0, w - new_w + 1, (1,))

        rgb = rgb[:,
                    top: top + new_h,
                    left: left + new_w]

        depth = depth[:,
                    top: top + new_h,
                    left: left + new_w]
        
        mask = mask[:,
                    top: top + new_h,
                    left: left + new_w]

        return (rgb, depth, mask)


