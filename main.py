from model.unet_model import UNet
from dataloader.drive_load import load_data, DRIVE_Dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm

if __name__ == '__main__':
    # Load data

    trn_x, trn_y, trn_mask, tst_x, tst_mask = load_data(path='/mnt/c/dataset')
    device = torch.device("cuda:0")
    train_dataset = DRIVE_Dataset(trn_x, trn_y, trn_mask)

    test_dataset = DRIVE_Dataset(tst_x, masks=tst_mask)    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    net = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.AdamW(net.parameters())
    
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    epochs = 10
    per_epoch_num = len(train_dataset) / 2
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            net.train()
            for (image, label, _) in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                loss = criterion(pred, label)
                # print('{}/{}：Loss/train'.format(epoch + 1, epochs), loss.item())
                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'best_model.pth')
                loss.backward()
                optimizer.step()
                pbar.update(1)
