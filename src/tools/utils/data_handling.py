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
        self.folder_path = folder_path
        self.transform = transform

        with h5py.File(self.folder_path, 'r') as f:
            self.total_images = f["images"].shape[0]
            
        self.class_count()

    def __getitem__(self, index):
        with h5py.File(self.folder_path, 'r') as f:
            rgb = torch.from_numpy(np.array(f["images"][index], dtype=np.uint8)).float()
            
            depth_raw = np.array(f["depths"][index], dtype=np.uint8)
            depth = torch.from_numpy(np.expand_dims(depth_raw, axis=0)).float()
            
            mask_raw = np.array(f["labels"][index], dtype=np.uint8)
            mask = torch.from_numpy(np.expand_dims(mask_raw, axis=0)).float()
            
        if self.transform:
            rgb, depth, mask = self.transform((rgb, depth, mask))
   
        return rgb, depth, mask

    def __len__(self):
        return self.total_images

    def class_count(self):
        with h5py.File(self.folder_path, 'r') as f:
            labels_tensor = torch.from_numpy(np.array(f["labels"], dtype=np.uint8))
            self.classes = torch.unique(labels_tensor).numpy()
            
        return self.classes
    
class SUNRGBDDataset1(Dataset):

    def __init__(self, root, num_classes=13, transform=None, img_size=(480, 640)):

        super(SUNRGBDDataset1, self).__init__()
        self.transform = transform

        samples = []
        for split in ['train', 'test']:
            list_file = os.path.join(root, f'{split}{num_classes}.txt')
            if not os.path.exists(list_file):
                print(f'Warning: {list_file} not found, skipping.')
                continue
            with open(list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) == 3:
                        rgb_path, depth_path, label_path = parts
                        samples.append((rgb_path, depth_path, label_path))

        print(f'Found {len(samples)} samples, loading into memory...')

        H, W = img_size
        N = len(samples)

        self.rgb_images   = torch.zeros((N, 3, H, W), dtype=torch.float32)
        self.depth_images = torch.zeros((N, 1, H, W), dtype=torch.float32)
        self.mask_images  = torch.zeros((N, 1, H, W), dtype=torch.float32)

        for i, (rgb_path, depth_path, label_path) in enumerate(samples):

            if i % 500 == 0:
                print(f'  Loading {i}/{N}...')

            # RGB (3, H, W)
            rgb = Image.open(os.path.join(root, rgb_path)).convert('RGB')
            rgb = rgb.resize((W, H))
            self.rgb_images[i] = torch.from_numpy(
                np.array(rgb, dtype=np.uint8)
            ).permute(2, 0, 1).float()

            # Depth (1, H, W)
            depth = Image.open(os.path.join(self.root, depth_path))
            depth = depth.resize((W, H))
            depth_np = np.array(depth, dtype=np.float32)
            depth_np = depth_np / depth_np.max() * 255.0 
            depth = torch.from_numpy(depth_np).unsqueeze(0).float()

            # Mask (1, H, W)
            mask = Image.open(os.path.join(root, label_path))
            mask = mask.resize((W, H), resample=Image.NEAREST)
            self.mask_images[i] = torch.from_numpy(
                np.array(mask, dtype=np.uint8)
            ).unsqueeze(0).float()

        print(f'Done. rgb: {self.rgb_images.shape}, '
              f'depth: {self.depth_images.shape}, '
              f'mask: {self.mask_images.shape}')

        self.class_count()

    def __len__(self):
        return self.rgb_images.shape[0]

    def __getitem__(self, index):
        rgb   = self.rgb_images[index]
        depth = self.depth_images[index]
        mask  = self.mask_images[index]

        if self.transform:
            rgb, depth, mask = self.transform((rgb, depth, mask))

        return rgb, depth, mask

    def class_count(self):
        self.classes = torch.unique(self.mask_images).numpy()
        print(f'Classes present in dataset: {self.classes}')
        return self.classes
    
class SUNRGBDDataset2(Dataset):

    def __init__(self, root, num_classes=13, transform=None, img_size=(480, 640)):

        super(SUNRGBDDataset2, self).__init__()
        self.root = root
        self.transform = transform
        self.img_size = img_size
        self.samples = []

        for split in ['train', 'test']:
            list_file = os.path.join(root, f'{split}{num_classes}.txt')
            if not os.path.exists(list_file):
                print(f'Warning: {list_file} not found, skipping.')
                continue
            with open(list_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) == 3:
                        rgb_path, depth_path, label_path = parts
                        self.samples.append((rgb_path, depth_path, label_path))

        print(f'Found {len(self.samples)} samples '
              f'images will be read from disk on demand)')

        self.classes = self._scan_classes()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        H, W = self.img_size
        rgb_path, depth_path, label_path = self.samples[index]

        # RGB (3, H, W)
        rgb = Image.open(os.path.join(self.root, rgb_path)).convert('RGB')
        rgb = rgb.resize((W, H))
        rgb = torch.from_numpy(
            np.array(rgb, dtype=np.uint8)
        ).permute(2, 0, 1).float()

        # Depth (1, H, W)
        depth = Image.open(os.path.join(self.root, depth_path))
        depth = depth.resize((W, H))
        depth_np = np.array(depth, dtype=np.float32)
        depth_np = depth_np / depth_np.max() * 255.0
        depth = torch.from_numpy(depth_np).unsqueeze(0).float()

        # Mask (1, H, W)
        mask = Image.open(os.path.join(self.root, label_path))
        mask = mask.resize((W, H), resample=Image.NEAREST)
        mask = torch.from_numpy(
            np.array(mask, dtype=np.uint8)
        ).unsqueeze(0).float()

        if self.transform:
            rgb, depth, mask = self.transform((rgb, depth, mask))

        return rgb, depth, mask

    def class_count(self):
        return self.classes

    def _scan_classes(self):

        print('Scanning mask files for unique classes '
              '(reads label PNGs only)...')
        unique = set()
        for i, (_, _, label_path) in enumerate(self.samples):
            if i % 500 == 0:
                print(f'  Scanning {i}/{len(self.samples)}...')
            mask = np.array(
                Image.open(os.path.join(self.root, label_path)),
                dtype=np.uint8
            )
            unique.update(np.unique(mask).tolist())

        classes = np.array(sorted(unique))
        print(f'Classes present in dataset: {classes}')
        return classes
    

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
    
class RandomRotation(object):

    def __init__(self, p=0.5):
        self.probability = p

    def __call__(self, sample):
        rgb, depth, mask = sample

        if torch.randint(100, size=(1,)) < self.probability * 100:

            angle = int(torch.randint(1, 4, size=(1,))) * 90

            rgb   = transforms.functional.rotate(rgb,   angle)
            depth = transforms.functional.rotate(depth, angle)

            mask  = transforms.functional.rotate(
                        mask, angle,
                        interpolation=transforms.InterpolationMode.NEAREST)

        return (rgb, depth, mask)
