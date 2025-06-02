import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image 


class Br35H(Dataset):
    def __init__(self, datadir, split='train'):
        self.datadir = datadir
        self.split = split
        if split == 'train':
            self.img_folder = 'Brain Tumour Br35H/images/TRAIN/'
            self.msk_folder = 'Brain Tumour Br35H/masks/'
        if split == 'val':
            self.img_folder = 'Brain Tumour Br35H/images/VAL/'
            self.msk_folder = 'Brain Tumour Br35H/masks/'
        elif split == 'test':
            self.img_folder = 'Brain Tumour Br35H/images/TEST/'
            self.msk_folder = 'Brain Tumour Br35H/masks/'

        self.image = []
        self.mask = []
        self.label = []  
        
        self._read_data()  

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
        ])

    def _read_data(self):
        image_dir = os.path.join(self.datadir, self.img_folder)
        # skip .csv 
        self.image = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if not img.endswith('.json')
        ]
        msk_dir = os.path.join(self.datadir, self.msk_folder)
        self.msk = [
            os.path.join(msk_dir, img)
            for img in os.listdir(image_dir)
            if not img.endswith('.json')
        ]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img_path = self.image[idx]
        msk_path = self.msk[idx]

        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        mask = np.array(Image.open(msk_path).convert('L'), dtype=np.uint8)

        mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image = Image.fromarray(image)  
            image = self.transform(image)

        return (image, mask)


