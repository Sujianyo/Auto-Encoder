import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image 


class DRIVE_Dataset(Dataset):
    def __init__(self, datadir, split='train'):
        self.datadir = datadir
        self.split = split
        if split in ['train', 'validation']:
            self.sub_folder = 'DRIVE/training/'
        elif split == 'test':
            self.sub_folder = 'DRIVE/test/'

        self.image = []
        self.mask = []
        self.label = []  
        
        self._read_data()  

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
        ])

    def _read_data(self):
        image_dir = os.path.join(self.datadir, self.sub_folder, 'images')
        self.image = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]

        mask_dir = os.path.join(self.datadir, self.sub_folder, 'mask')
        self.mask = [os.path.join(mask_dir, img) for img in os.listdir(mask_dir)]

        if self.split == 'train':
            label_dir = os.path.join(self.datadir, self.sub_folder, '1st_manual')
            self.label = [os.path.join(label_dir, img) for img in os.listdir(label_dir)]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img_path = self.image[idx]
        msk_path = self.mask[idx]

        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        mask = np.array(Image.open(msk_path).convert('L'), dtype=np.uint8)

        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.split == 'train':
            lbl_path = self.label[idx]
            label = np.array(Image.open(lbl_path).convert('L'), dtype=np.uint8)
            label = (label.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image = Image.fromarray(image)  
            image = self.transform(image)

        return (image, label, mask) if self.split == 'train' else (image, mask)
