import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image  # 处理 GIF 图像

def load_data(dataset='drive', path='/mnt/c/dataset'):

    if dataset == 'drive':
        dataset = 'DRIVE'
        trn_path = os.path.join(path, dataset, 'training')
        tst_path = os.path.join(path, dataset, 'test')
        
        trn_x, trn_y, trn_mask = [], [], []   
        tst_x, tst_mask = [], []  

        for i in os.listdir(trn_path):
            pt = os.path.join(trn_path, i)
            if i == '1st_manual': 
                for j in os.listdir(pt):
                    img_path = os.path.join(pt, j)
                    img = Image.open(img_path).convert("L") if j.endswith('.gif') else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    trn_y.append(np.array(img, dtype=np.uint8))
            elif i == 'images':  
                for j in os.listdir(pt):
                    img_path = os.path.join(pt, j)
                    img = Image.open(img_path).convert("RGB") if j.endswith('.gif') else cv2.imread(img_path)
                    trn_x.append(np.array(img, dtype=np.uint8))
            else:  
                for j in os.listdir(pt):
                    img_path = os.path.join(pt, j)
                    img = Image.open(img_path).convert("L") if j.endswith('.gif') else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    trn_mask.append(np.array(img, dtype=np.uint8))

        for i in os.listdir(tst_path):
            pt = os.path.join(tst_path, i)
            if i == 'images':  
                for j in os.listdir(pt):
                    img_path = os.path.join(pt, j)
                    img = Image.open(img_path).convert("RGB") if j.endswith('.gif') else cv2.imread(img_path)
                    tst_x.append(np.array(img, dtype=np.uint8))
            else:  
                for j in os.listdir(pt):
                    img_path = os.path.join(pt, j)
                    img = Image.open(img_path).convert("L") if j.endswith('.gif') else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    tst_mask.append(np.array(img, dtype=np.uint8))

    return trn_x, trn_y, trn_mask, tst_x, tst_mask


class DRIVE_Dataset(Dataset):
    def __init__(self, images, labels=None, masks=None, transform=None):
        self.images = images
        self.labels = labels if labels is not None else [None] * len(images)  
        self.masks = masks
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),   
            # transforms.Resize((256, 256)),  
            transforms.ToTensor(),  
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]  
        mask = self.masks[idx]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if isinstance(image, np.ndarray) else image
        mask = mask if isinstance(mask, np.ndarray) else np.array(mask)

        if label is not None:
            label = label if isinstance(label, np.ndarray) else np.array(label)


        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)  
        mask = torch.tensor(mask).unsqueeze(0)

        if label is not None:
            label = np.array(label, dtype=np.float32) / 255.0
            label = (label > 0.5).astype(np.float32)  
            label = torch.tensor(label).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
        # print(image.size(), label.size(), mask.size())
        return (image, label, mask) if label is not None else (image, mask)  

