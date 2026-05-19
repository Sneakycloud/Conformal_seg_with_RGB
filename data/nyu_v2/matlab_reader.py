from os.path import dirname, join as pjoin
import scipy.io as sio
import os
import h5py
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

data_dir =  os.path.dirname(os.path.realpath(__file__))
mat_fname = pjoin(data_dir, 'nyu_depth_v2_labeled.mat')
#mat_contents = sio.loadmat(mat_fname, spmatrix=False)
#print(sorted(mat_contents.keys()))

f = h5py.File(mat_fname,'r')
print(sorted(f.keys()))


#matplotlib setup
fig, axs = plt.subplots(3)
fig.suptitle("Nyu v2 images")

#rgb image
rgb = f["images"]
print(rgb)
rgb_2d_list = rgb[0]
print(rgb_2d_list)
rgb_np = np.array(rgb_2d_list,dtype=np.uint8)
rgb_np = np.transpose(rgb_np, (2,1,0))
print(rgb_np.shape)
rgb_image = Image.fromarray(rgb_np, "RGB")

axs[0].imshow(rgb_image)

#depth image
depth_images = f["depths"]
print(depth_images)
depth_image_2d_list = depth_images[0]
print(depth_image_2d_list)
depth_np = np.array(depth_image_2d_list,dtype=np.uint8)
depth_np = np.transpose(depth_np, (1,0))
depth_image = Image.fromarray(depth_np)

axs[1].imshow(depth_image)

#semantic mask
mask_images = f["labels"]
print(mask_images)
mask_image_2d_list = mask_images[0]
print(mask_image_2d_list)
mask_np = np.array(mask_image_2d_list,dtype=np.uint8)
mask_np = np.transpose(mask_np, (1,0))
mask_image = Image.fromarray(mask_np)

axs[2].imshow(mask_image)

#Show
plt.show()