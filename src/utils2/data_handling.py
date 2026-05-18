import os
import cv2
import logging
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

class RandomFlip:
    """Randomly applies a horizontal/vertical flip to inputs and target simultaneously"""
    def __init__(self, rng):
        self._rng = rng

    @classmethod
    def _flip_image(cls, image, flip_code):
        if flip_code is None:
            return image
        return cv2.flip(image, flip_code)

    def __call__(self, sample):
        flip_code = self._rng.choice([0, 1, -1, None])
        transformed_inputs = []
        for original_input in sample['inputs']:
            transformed_inputs.append(self._flip_image(original_input, flip_code))

        transformed_target = self._flip_image(sample['target'], flip_code)
        return {'inputs': transformed_inputs, 'target': transformed_target}


class RandomRotate:
    """Randomly rotates inputs and target simultaneously by 90, 180, or 270 degrees"""
    def __init__(self, rng):
        self._rng = rng

    @classmethod
    def _rotate_image(cls, image, code):
        if code is None:
            return image
        return cv2.rotate(image, code)

    def __call__(self, sample):
        code = self._rng.choice([cv2.ROTATE_90_CLOCKWISE, 
                                 cv2.ROTATE_180, 
                                 cv2.ROTATE_90_COUNTERCLOCKWISE, 
                                 None])
        transformed_inputs = []
        for original_input in sample['inputs']:
            transformed_inputs.append(self._rotate_image(original_input, code))

        transformed_target = self._rotate_image(sample['target'], code)
        return {'inputs': transformed_inputs, 'target': transformed_target}


class ToTensor:
    """Converts numpy ndarrays in sample to PyTorch Tensors with correct dimensions"""
    def __call__(self, sample):
        torch_inputs = []
        for img in sample['inputs']:
            # Handle single channel (H, W) depth maps vs (H, W, C) RGB images safely
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2) # Convert (H, W) to (H, W, 1)
            
            # Change memory layout from HWC to CHW format expected by PyTorch
            torch_inputs.append(torch.from_numpy(img.transpose((2, 0, 1))).float())
        
        # Ground truth targets MUST be integers (long) for PyTorch CrossEntropyLoss
        target = torch.from_numpy(sample['target']).long()
        
        return {'inputs': torch_inputs, 'target': target}


class MultibandDataset(Dataset):
    """
    Flexible dataset loader capable of managing single or multi-view streams
    (e.g., loading an RGB directory stream alongside a Depth directory stream).
    """
    def __init__(self, img_paths, classes, id_file, gt_path=None, transform=None):
        """
        Parameters
        ----------
        img_paths : List of folder paths (e.g., ['/path/to/rgb', '/path/to/depth'])
        classes   : List of raw class integer labels (e.g., [0, 1, 2, 3...])
        id_file   : TXT file containing specific image IDs for this CV split fold
        """
        self.img_paths = img_paths
        self.classes = classes
        self.gt_path = gt_path
        self.transform = transform

        # Load file names/IDs assigned to this cross-validation fold split
        with open(id_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        inputs = []

        # Gather arrays across all provided modality directories (RGB, Depth, etc.)
        for folder in self.img_paths:
            # Assumes files share structural names across folders (e.g., '0001.png')
            full_path = os.path.join(folder, f"{img_id}.png")
            # Read unchanged to preserve raw depth float/integer values
            img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Could not read input file: {full_path}")
            inputs.append(img.astype(np.float32))

        # Target mask retrieval
        target = None
        if self.gt_path:
            gt_full_path = os.path.join(self.gt_path, f"{img_id}.png")
            target = cv2.imread(gt_full_path, cv2.IMREAD_UNCHANGED)
            if target is None:
                raise FileNotFoundError(f"Could not read ground truth file: {gt_full_path}")

        sample = {'inputs': inputs, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def calculate_median_frequency_weights(self):
        """ Computes class balancing weights via Median Frequency Balancing (MFB) """
        logging.info("Calculating Median Frequency Balancing Weights...")
        class_counts = {c: 0 for c in self.classes}
        class_totals = {c: 0 for c in self.classes}

        for img_id in self.ids:
            gt_full_path = os.path.join(self.gt_path, f"{img_id}.png")
            target = cv2.imread(gt_full_path, cv2.IMREAD_UNCHANGED)
            target_pixels = np.prod(target.shape)
            
            for cls_idx in self.classes:
                class_counts[cls_idx] += np.sum(target == cls_idx)
                if (target == cls_idx).any():
                    class_totals[cls_idx] += target_pixels

        frequencies = []
        for cls_idx in self.classes:
            cnt = class_counts[cls_idx]
            tot = class_totals[cls_idx]
            if cnt == 0 and tot == 0:
                frequencies.append(0.0001)  # Stability fallback for completely unrepresented classes
            else:
                frequencies.append(cnt / tot)

        median_freq = np.median(frequencies)
        weights = [median_freq / freq for freq in frequencies]
        return np.array(weights, dtype=np.float32)