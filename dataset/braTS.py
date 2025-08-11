import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd
import random
from albumentations.pytorch import ToTensorV2
class BraTSDataset(Dataset):
    def __init__(self, root_dir, modalities=['flair', 't1', 't1ce', 't2'],
                 filter_empty_mask=True, transforms=None,
                 label_mode="merged",  # "merged" or "multi"
                 selected_labels=(1, 2, 4)):
        """
        Args:
            root_dir (str): dataset folder path
            modalities (list): MRI modalities to load
            filter_empty_mask (bool): skip slices with no foreground
            transforms: Albumentations transforms
            label_mode (str): "merged" → one binary mask for all selected labels
                              "multi"  → one channel per selected label
            selected_labels (tuple): labels to use (BraTS labels: 1, 2, 4)
        """
        self.root_dir = root_dir
        self.modalities = modalities
        self.filter_empty_mask = filter_empty_mask
        self.transforms = transforms
        self.label_mode = label_mode
        self.selected_labels = tuple(selected_labels)
        self.samples = []

        self.patient_dirs = sorted([f for f in os.listdir(root_dir) if f.startswith('BraTS20')])

        for patient_id in self.patient_dirs:
            try:
                volume_paths = {
                    mod: os.path.join(root_dir, patient_id, f"{patient_id}_{mod}.nii")
                    for mod in self.modalities
                }
                mask_path = os.path.join(root_dir, patient_id, f"{patient_id}_seg.nii")

                if not os.path.exists(mask_path) or any(not os.path.exists(vp) for vp in volume_paths.values()):
                    print(f"[SKIP] Missing file(s) for patient: {patient_id}")
                    continue

                flair_data = nib.load(volume_paths['flair']).get_fdata()
                mask_data = nib.load(mask_path).get_fdata()
                depth = flair_data.shape[2]

                for z in range(depth):
                    mask_slice = mask_data[:, :, z]
                    if self.filter_empty_mask and np.max(mask_slice) == 0:
                        continue
                    self.samples.append((patient_id, z))

            except Exception as e:
                print(f"[ERROR] Failed to load patient {patient_id}: {e}")
                continue

        # Optional debug: limit dataset size
        # self.samples = self.samples[:100]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, z = self.samples[idx]

        # Load image modalities
        volume = []
        for mod in self.modalities:
            path = os.path.join(self.root_dir, patient_id, f"{patient_id}_{mod}.nii")
            data = nib.load(path).get_fdata()
            slice_2d = data[:, :, z]

            mean, std = slice_2d.mean(), slice_2d.std()
            if std == 0:
                std = 1
            norm_slice = (slice_2d - mean) / std
            volume.append(norm_slice)

        image = np.stack(volume, axis=0).astype(np.float32)  # [C, H, W]

        # Load mask
        mask_path = os.path.join(self.root_dir, patient_id, f"{patient_id}_seg.nii")
        mask_data = nib.load(mask_path).get_fdata()
        mask_slice = mask_data[:, :, z].astype(np.uint8)  # [H, W]

        # Process mask depending on mode
        if self.label_mode == "merged":
            # All selected labels merged into one binary mask
            mask_slice = np.isin(mask_slice, self.selected_labels).astype(np.uint8)
        elif self.label_mode == "multi":
            # One channel per label
            mask_multi = [(mask_slice == lbl).astype(np.uint8) for lbl in self.selected_labels]
            mask_slice = np.stack(mask_multi, axis=0)  # [num_labels, H, W]
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        # Apply transforms (Albumentations expects HWC for image, [H, W] for mask)
        if self.transforms:
            image_hwc = np.transpose(image, (1, 2, 0))  # [H, W, C]

            if self.label_mode == "multi":
                # Albumentations doesn't support multi-channel masks directly
                # Apply transform to each channel separately
                mask_aug_list = []
                for c in range(mask_slice.shape[0]):
                    augmented = self.transforms(image=image_hwc, mask=mask_slice[c])
                    if c == 0:
                        image_tensor = augmented['image']
                    mask_aug_list.append(augmented['mask'])
                mask_tensor = torch.stack(mask_aug_list, dim=0).float()
            else:
                augmented = self.transforms(image=image_hwc, mask=mask_slice)
                image_tensor = augmented['image']
                mask_tensor = augmented['mask'].float()
        else:
            image_tensor = torch.from_numpy(image).float()
            if self.label_mode == "multi":
                mask_tensor = torch.from_numpy(mask_slice).float()
            else:
                mask_tensor = torch.from_numpy(mask_slice).float()

        return image_tensor, mask_tensor
