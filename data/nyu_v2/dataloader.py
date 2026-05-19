from os.path import dirname, join as pjoin
import scipy.io as sio
import os
import h5py
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image

class RGBD_Segmentation_Dataset(Dataset):
    def __init__(self, folder_path, transform):
        super(RGBD_Segmentation_Dataset, self).__init__()
        with h5py.File(folder_path,'r') as f:
            self.rgb_images = torch.from_numpy(np.array(f["images"],dtype=np.uint8)).float() # N x C x H x W
            self.depth_images = torch.from_numpy(np.expand_dims(np.array(f["depths"],dtype=np.uint8),axis=1)).float() #The expand dims is to get N x C x H x W
            self.mask_images = torch.from_numpy(np.expand_dims(np.array(f["labels"],dtype=np.uint8),axis=1)).float() #The expand dims is to get N x C x H x W
        self.transform = transform

    def __getitem__(self, index):
            rgb = self.rgb_images[index]
            depth = self.depth_images[index]
            mask = self.mask_images[index]
            
            if self.transform:
                rgb, depth, mask = self.transform((rgb, depth, mask))    
            
            return rgb, depth, mask

    def __len__(self):
        return self.rgb_images.shape[0]

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



#loads data
data_dir =  os.path.dirname(os.path.realpath(__file__))
mat_fname = pjoin(data_dir, 'nyu_depth_v2_labeled.mat')

transform_compose = transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip()])
#transform_compose = transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),RandomCrop(300)])

semantic_dataset = RGBD_Segmentation_Dataset(mat_fname, transform_compose)
RGBD_dataloader = DataLoader(semantic_dataset, batch_size=4, shuffle=True)

for i_batch, (rgb, depth, mask) in enumerate(RGBD_dataloader):
    #rgb = (num of images in batch,channel,height,width)
    print(f"Batch {i_batch}")
    print(rgb.shape)
    print(depth.shape)
    print(mask.shape)
    print("\n")
    
    #matplotlib setup
    fig, axs = plt.subplots(3)
    fig.suptitle("Nyu v2 images")
    
    rgb_np = np.uint8(rgb[0].numpy())
    #print(f"rgb np: {rgb_np.shape}")
    rgb_np = np.transpose(rgb_np, (1,2,0))
    rgb_image = Image.fromarray(rgb_np, "RGB")
    axs[0].imshow(rgb_image)
    
    depth_np = np.uint8(depth[0].squeeze(0).numpy())
    depth_np = np.transpose(depth_np, (0,1))
    depth_image = Image.fromarray(depth_np)
    axs[1].imshow(depth_image)

    mask_np = np.uint8(mask[0].squeeze(0).numpy())
    depth_np = np.transpose(depth_np, (0,1))
    mask_image = Image.fromarray(mask_np)
    axs[2].imshow(mask_image)
    
    plt.show()
    
    if i_batch == 0:
        break
