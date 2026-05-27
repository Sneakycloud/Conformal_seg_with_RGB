import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold

from data_handling import SUNRGBDDataset1
from data_handling import SUNRGBDDataset2

def test_dataset(dataset, name):

    print(f'\n{"="*60}')
    print(f'Testing: {name}')
    print(f'{"="*60}')

    print(f'\n[1] Dataset length: {len(dataset)}')
    assert len(dataset) > 0, 'Dataset is empty'

    rgb, depth, mask = dataset[0]
    print(f'\n[2] Single sample shapes:')
    print(f'    rgb   : {rgb.shape}   (expected: (3, H, W))')
    print(f'    depth : {depth.shape}  (expected: (1, H, W))')
    print(f'    mask  : {mask.shape}  (expected: (1, H, W))')
    assert rgb.shape[0]   == 3, f'RGB should have 3 channels, got {rgb.shape[0]}'
    assert depth.shape[0] == 1, f'Depth should have 1 channel, got {depth.shape[0]}'
    assert mask.shape[0]  == 1, f'Mask should have 1 channel, got {mask.shape[0]}'
    assert rgb.shape[1:] == depth.shape[1:] == mask.shape[1:], \
        'RGB, depth and mask spatial dimensions do not match'
    print('    All shapes correct')

    print(f'\n[3] Dtypes:')
    print(f'    rgb   : {rgb.dtype}   (expected: torch.float32)')
    print(f'    depth : {depth.dtype}  (expected: torch.float32)')
    print(f'    mask  : {mask.dtype}  (expected: torch.float32)')
    assert rgb.dtype   == torch.float32, 'RGB dtype should be float32'
    assert depth.dtype == torch.float32, 'Depth dtype should be float32'
    assert mask.dtype  == torch.float32, 'Mask dtype should be float32'
    print('    All dtypes correct')

    print(f'\n[4] Value ranges:')
    print(f'    rgb   : [{rgb.min():.1f}, {rgb.max():.1f}]   (expected: 0-255)')
    print(f'    depth : [{depth.min():.1f}, {depth.max():.1f}]  (expected: 0-255)')
    print(f'    mask  : [{mask.min():.1f}, {mask.max():.1f}]  (expected: 0-num_classes)')
    assert rgb.min() >= 0 and rgb.max() <= 255,     'RGB values out of range'
    assert depth.min() >= 0 and depth.max() <= 255, 'Depth values out of range'
    print('    Value ranges correct')

    print(f'\n[5] Classes in dataset:')
    classes = dataset.class_count()
    print(f'    {classes}')
    assert len(classes) > 0, 'No classes found'

    print(f'\n[6] DataLoader batch check (batch_size=4):')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    rgb_batch, depth_batch, mask_batch = next(iter(loader))
    print(f'    rgb_batch   : {rgb_batch.shape}   (expected: (4, 3, H, W))')
    print(f'    depth_batch : {depth_batch.shape}  (expected: (4, 1, H, W))')
    print(f'    mask_batch  : {mask_batch.shape}  (expected: (4, 1, H, W))')
    assert rgb_batch.shape   == (4, 3, *rgb.shape[1:]), 'Batch RGB shape wrong'
    assert depth_batch.shape == (4, 1, *rgb.shape[1:]), 'Batch depth shape wrong'
    assert mask_batch.shape  == (4, 1, *rgb.shape[1:]), 'Batch mask shape wrong'
    print('    Batch shapes correct')

    print(f'\n[7] KFold split check (5 folds for speed):')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=4)
        val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=4)
        print(f'    Fold {fold+1}: train={len(train_idx)} samples, '
              f'val={len(val_idx)} samples, '
              f'train batches={len(train_loader)}, '
              f'val batches={len(val_loader)}')
    print('    KFold splits correct')

    print(f'\n[8] Saving visual check to {name}_visual_check.png ...')
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'{name} — sample 0')

    rgb_np = np.uint8(rgb.numpy().transpose(1, 2, 0))
    axs[0].imshow(rgb_np)
    axs[0].set_title('RGB')
    axs[0].axis('off')

    depth_np = np.uint8(depth.squeeze(0).numpy())
    axs[1].imshow(depth_np, cmap='plasma')
    axs[1].set_title('Depth')
    axs[1].axis('off')

    mask_np = np.uint8(mask.squeeze(0).numpy())
    axs[2].imshow(mask_np, cmap='tab20')
    axs[2].set_title('Segmentation mask')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{name}_visual_check.png', dpi=150)
    plt.close()
    print(f'    Saved!')

    print(f'\n✓ All checks passed for {name}')



if __name__ == '__main__':

    SUNRGBD_ROOT = '/home/vencilo/School/Deep_Learning/SUN_RGBD'

    print('Initialising dataloader2...')
    dataloader2 = SUNRGBDDataset2(
        root=SUNRGBD_ROOT,
        num_classes=37,
        transform=None,
        img_size=(480, 640)
    )
    test_dataset(dataloader2, 'Dataloader2')

    RUN_PRELOADED_TEST = False
    if RUN_PRELOADED_TEST:
        print('\nInitialising dataloader1...')
        dataloader1 = SUNRGBDDataset1(
            root=SUNRGBD_ROOT,
            num_classes=37,
            transform=None,
            img_size=(480, 640)
        )
        test_dataset(dataloader1, 'Dataloader1')