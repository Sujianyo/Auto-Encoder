import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd
import random
from albumentations.pytorch import ToTensorV2

# class BraTSSliceDataset(Dataset):
#     def __init__(self, root_dir, name_map_csv, survival_csv, image_size=(128, 128), slice_range=(30, 130), num_subjects_to_use=20):
#         self.root_dir = root_dir
#         self.image_size = image_size
#         # self.slice_range = slice_range
#         self.modalities = ['flair', 't1', 't1ce', 't2']
        
#         # Load CSVs
#         self.name_map = pd.read_csv(name_map_csv)
#         self.survival = pd.read_csv(survival_csv)
        
#         # self.subjects = sorted([
#         #     d for d in os.listdir(root_dir)
#         #     if d.startswith("BraTS20_Training")
#         # ])
        
#         # Get list of subject directories
#         all_subjects = []
#         for d in sorted(os.listdir(root_dir)):
#             subj_path = os.path.join(root_dir, d)
#             if not os.path.isdir(subj_path):
#                 continue
        
#             try:
#                 # Check if all 5 required files exist
#                 required = [f"{d}_{mod}.nii" for mod in self.modalities] + [f"{d}_seg.nii"]
#                 if all(os.path.exists(os.path.join(subj_path, fname)) for fname in required):
#                     all_subjects.append(d)
#             except Exception as e:
#                 print(f"> Skipping {d} due to error: {e}")

#         # Limit to specified number of subjects
#         if num_subjects_to_use is not None:
#             print("num_subjects_to_use =", num_subjects_to_use)
#             random.seed(42)  # for reproducibility
#             self.subjects = random.sample(all_subjects, num_subjects_to_use)
#         else:
#             self.subjects = all_subjects

#         # for sid in self.subjects:
#         #     for i in range(*slice_range):
#         #         self.slices.append((sid, i))
        
#         # Prepare slice list
#         self.slices = []  # (subject_id, slice_index)
#         for sid in self.subjects:
#             any_mod_path = os.path.join(self.root_dir, sid, f"{sid}_flair.nii")
#             vol = nib.load(any_mod_path).get_fdata()
#             num_slices = vol.shape[2]
            
#             for i in range(num_slices):
#                 self.slices.append((sid, i))
#         self.aug_transform = A.Compose([
#             A.Resize(width=self.image_size[0], height=self.image_size[1]),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.5),
#             A.RandomRotate90(p=0.5),
#             A.Transpose(p=0.5),
#             A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
#             A.Normalize(),  # This uses default mean=[0,0,0] and std=[1,1,1] unless otherwise specified
#             ToTensorV2(),
#         ])
#         # self.transform = T.Compose([
#         #     T.Resize(image_size),
#         #     T.ToTensor(),
#         # ])

#     def __len__(self):
#         return len(self.slices)

#     def __getitem__(self, idx):
#         subject_id, slice_idx = self.slices[idx]
#         path = os.path.join(self.root_dir, subject_id)

#         # Load modalities and stack into a 3D array (H x W x C)
#         slices = []
#         for mod in self.modalities:
#             img_path = os.path.join(path, f"{subject_id}_{mod}.nii")
#             vol = nib.load(img_path).get_fdata()
#             img_slice = vol[:, :, slice_idx]
            
#             # Z-score normalization
#             mean = np.mean(img_slice)
#             std = np.std(img_slice)
#             img_slice = (img_slice - mean) / (std + 1e-5)
            
#             slices.append(img_slice)

#         # Stack to shape: (H, W, 4) for Albumentations
#         image = np.stack(slices, axis=-1).astype(np.float32)

#         # Load segmentation
#         seg_path = os.path.join(path, f"{subject_id}_seg.nii")
#         seg_vol = nib.load(seg_path).get_fdata()
#         seg_slice = seg_vol[:, :, slice_idx].astype(np.uint8)

#         # Apply augmentations
#         transformed = self.aug_transform(image=image, mask=seg_slice)
#         image_tensor = transformed["image"]  # shape: (C, H, W)
#         seg_tensor = transformed["mask"].long()  # shape: (H, W)

#         # Load metadata
#         grade_row = self.name_map[self.name_map['BraTS_2020_subject_ID'] == subject_id]
#         grade = grade_row['Grade'].values[0] if not grade_row.empty else "Unknown"

#         survival_row = self.survival[self.survival['Brats20ID'] == subject_id]
#         survival_days = survival_row['Survival_days'].values[0] if not survival_row.empty else -1

#         return {
#             'image': image_tensor,       # (4, H, W)
#             'seg': seg_tensor,           # (H, W)
#             'subject_id': subject_id,
#             'slice_idx': slice_idx,
#             'grade': grade,
#             'survival': survival_days
#         }
    
class BraTSDataset(Dataset):
    def __init__(self, root_dir, modalities=['flair', 't1', 't1ce', 't2'],
                 filter_empty_mask=True, transforms=None):
        self.root_dir = root_dir
        self.modalities = modalities
        self.filter_empty_mask = filter_empty_mask
        self.transforms = transforms
        self.samples = []  # List of (patient_id, slice_index)

        # Get list of patient folders
        self.patient_dirs = sorted([f for f in os.listdir(root_dir) if f.startswith('BraTS20')])

        for patient_id in self.patient_dirs:
            try:
                volume_paths = {
                    mod: os.path.join(root_dir, patient_id, f"{patient_id}_{mod}.nii")
                    for mod in self.modalities
                }
                mask_path = os.path.join(root_dir, patient_id, f"{patient_id}_seg.nii")

                # Check if all required files exist
                if not os.path.exists(mask_path) or any(not os.path.exists(vp) for vp in volume_paths.values()):
                    print(f"[SKIP] Missing file(s) for patient: {patient_id}")
                    continue

                # Load sample flair + mask to get number of slices
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, z = self.samples[idx]

        volume = []
        for mod in self.modalities:
            path = os.path.join(self.root_dir, patient_id, f"{patient_id}_{mod}.nii")
            data = nib.load(path).get_fdata()
            slice_2d = data[:, :, z]

            # Z-score normalization
            mean = np.mean(slice_2d)
            std = np.std(slice_2d)
            if std == 0:
                std = 1
            norm_slice = (slice_2d - mean) / std
            volume.append(norm_slice)

        # Stack modalities into 4-channel image [C, H, W]
        image = np.stack(volume, axis=0).astype(np.float32)  # shape [4, H, W]

        # Load mask
        mask_path = os.path.join(self.root_dir, patient_id, f"{patient_id}_seg.nii")
        mask_data = nib.load(mask_path).get_fdata()
        mask_slice = mask_data[:, :, z].astype(np.uint8)    # shape [H, W]

        # Albumentations  HWC 
        if self.transforms:
            image = np.transpose(image, (1, 2, 0))  # [H, W, C]
            augmented = self.transforms(image=image, mask=mask_slice)
            image_tensor = augmented['image']    # numpy array
            mask_tensor = augmented['mask']
        else:
            image_tensor = torch.from_numpy(image).float()
            mask_tensor = torch.from_numpy(mask_slice).long()
        return image_tensor, mask_tensor