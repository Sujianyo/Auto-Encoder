import cv2
import torch
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# lgg dataset
def plot_to_tensorboard(n_examples, list_img_paths, list_mask_paths, writer: SummaryWriter, global_step=0, tag_name='MRI_Images'):
    for i in range(n_examples):
        img_path = list_img_paths[i]
        mask_path = list_mask_paths[i]

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        highlighted = image.copy()
        overlay = mask.copy()
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, highlighted, 1 - alpha, 0, highlighted)

        def to_tensor(img):
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            return img

        img_tensor = to_tensor(image)
        highlighted_tensor = to_tensor(highlighted)
        mask_tensor = to_tensor(mask)

        grid = make_grid([img_tensor, highlighted_tensor, mask_tensor], nrow=3)

        # All images go under the same tag, differentiated by step
        writer.add_image(tag_name+f'/{i}', grid, global_step=global_step + i)

import random
import torch
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

def visualize_segmentation_tensorboard(model, dataloader, device, writer: SummaryWriter, num_examples=4):
    model.eval()

    dataset_size = len(dataloader.dataset)
    random_indices = random.sample(range(dataset_size), num_examples)

    for i, idx in enumerate(random_indices):
        img, mask = dataloader.dataset[idx]
        img, mask = img.to(device).unsqueeze(0), mask.to(device).unsqueeze(0)

        with torch.no_grad():
            pred = model(img)

        # Prepare image tensors for TensorBoard
        img_np = img.squeeze(0).cpu()
        if img_np.shape[0] == 1:
            img_disp = img_np.repeat(3, 1, 1)
        else:
            img_disp = img_np[:3]  # limit to first 3 channels

        # Normalize image to 0-1 for display
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

        # Process masks
        mask_np = mask.squeeze().cpu().float().numpy()
        pred_np = pred.squeeze().cpu().float().numpy()
        binary_pred_np = (pred_np > 0.5).astype(np.float32)

        def to_rgb_tensor(np_img):
            if np_img.ndim == 2:
                tensor = torch.tensor(np_img).unsqueeze(0).float()
            else:
                tensor = torch.tensor(np_img).float()
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
            return tensor.repeat(3, 1, 1)

        mask_tensor = to_rgb_tensor(mask_np)
        pred_tensor = to_rgb_tensor(pred_np)
        binary_pred_tensor = to_rgb_tensor(binary_pred_np)

        grid = make_grid(
            [img_disp, mask_tensor, pred_tensor, binary_pred_tensor],
            nrow=4,
            normalize=False  # Already normalized manually
        )

        writer.add_image(f"Segmentation/{idx}", grid, global_step=i)

def show_aug(datas, writer: SummaryWriter, norm=False, num_examples=4):
    dataset_size = len(datas.dataset)
    random_indices = random.sample(range(dataset_size), num_examples)

    for i, idx in enumerate(random_indices):
        img, mask = datas.dataset[idx]
        img_np = img.squeeze(0).cpu()
        if img_np.shape[0] == 1:
            img_disp = img_np.repeat(3, 1, 1)
        else:
            img_disp = img_np[:3]  # limit to first 3 channels
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
        mask_np = mask.squeeze().cpu().float().numpy()
        def to_rgb_tensor(np_img):
            if np_img.ndim == 2:
                tensor = torch.tensor(np_img).unsqueeze(0).float()
            else:
                tensor = torch.tensor(np_img).float()
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
            return tensor.repeat(3, 1, 1)
        # if norm:           
        #     img = inputs[idx].numpy().transpose(1,2,0)
        #     mean = [0.485, 0.456, 0.406]
        #     std = [0.229, 0.224, 0.225] 
        #     img = (img*std+mean).astype(np.float32)
            
        # else:
        #     img = inputs[idx].numpy().astype(np.float32)
        #     img = img[0,:,:]
        mask_tensor = to_rgb_tensor(mask_np)
        grid = make_grid(
            [img_disp, mask_tensor],
            nrow=2,
            normalize=False  # Already normalized manually
        )
        writer.add_image(f"Augment/{idx}", grid, global_step=i)
