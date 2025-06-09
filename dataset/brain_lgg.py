import os
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np
import cv2
class MRI_Dataset(Dataset):
    def __init__(self, df, img_transform=None):
        self.image_paths = df['image']
        self.mask_paths = df['mask']
        self.transforms = img_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_file = self.image_paths.iloc[idx]
        mask_file = self.mask_paths.iloc[idx]

        # img = np.array(Image.open(img_file).convert('RGB', dtype=np.unit8))
        # img = np.array(Image.open(mask_file).convert('L', dtype=np.unit8))

        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        # if self.img_transform:
        #     img = self.img_transform(img)
        # if self.mask_transform:
        #     mask = self.mask_transform(mask)
        augmented = self.transforms(image=img,
                                   mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        
        return image, mask
    
