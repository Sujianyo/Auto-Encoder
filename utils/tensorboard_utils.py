import cv2
import torch
import numpy as np
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

# lgg dataset
@torch.no_grad()
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
@torch.no_grad()
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
        img_np = img.squeeze(0)
        if img_np.shape[0] == 1:
            img_disp = img_np.repeat(3, 1, 1)
        else:
            img_disp = img_np[:3]  # limit to first 3 channels

        # Normalize image to 0-1 for display
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

        # Process masks
        mask_np = mask.squeeze().float().numpy()
        pred_np = pred.squeeze().float().numpy()
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
@torch.no_grad()
def show_aug(datas, writer: SummaryWriter, norm=False, num_examples=4):
    dataset_size = len(datas.dataset)
    random_indices = random.sample(range(dataset_size), num_examples)

    for i, idx in enumerate(random_indices):
        img, mask = datas.dataset[idx]
        img_np = img.squeeze(0)
        if img_np.shape[0] == 1:
            img_disp = img_np.repeat(3, 1, 1)
        else:
            img_disp = img_np[:3]  # limit to first 3 channels
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
        mask_np = mask.squeeze().float().numpy()
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
import torch
import random
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

@torch.no_grad()
def show_aug2(dataloader, writer: SummaryWriter, num_examples=4):
    """
    可视化数据增强后的图像和掩码到 TensorBoard。
    """
    data_iter = iter(dataloader)
    images, masks = next(data_iter)  # 一个 batch

    batch_size = images.size(0)
    num_examples = min(num_examples, batch_size)

    for i in range(num_examples):
        img = images[i]  # [C, H, W]
        mask = masks[i]  # [3, H, W]  -> WT, TC, ET

        # 只展示前三个通道作为图像（通常为 T1, T1c, T2）
        img_disp = img[:3]
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

        # 处理 mask，生成彩色图
        def to_rgb_mask(mask_tensor):
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.tensor(mask_tensor)

            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]

            mask_tensor = mask_tensor.float()
            mask_tensor = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min() + 1e-8)

            if mask_tensor.size(0) == 1:
                return mask_tensor.repeat(3, 1, 1)
            elif mask_tensor.size(0) >= 3:
                return mask_tensor[:3]
            else:
                raise ValueError(f"Unexpected mask shape: {mask_tensor.shape}")
        
        mask_rgb = to_rgb_mask(mask)
        print(mask[0].unique(), mask[1].unique(), mask[2].unique())
        # 确保尺寸一致
        if img_disp.shape != mask_rgb.shape:
            print(f"[Warning] Size mismatch: img {img_disp.shape}, mask {mask_rgb.shape}")
            min_h = min(img_disp.shape[1], mask_rgb.shape[1])
            min_w = min(img_disp.shape[2], mask_rgb.shape[2])
            img_disp = img_disp[:, :min_h, :min_w]
            mask_rgb = mask_rgb[:, :min_h, :min_w]

        # 拼图并写入 TensorBoard
        grid = make_grid(
            [img_disp, mask_rgb],
            nrow=2,
            normalize=False
        )
        writer.add_image(f"Augment/{i}", grid, global_step=0)


import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

@torch.no_grad()
def log_model_output_to_tensorboard(images, outputs, masks, writer: SummaryWriter, global_step=0, max_samples=2):
    """
    将模型输入、输出、标签的各通道灰度图保存到 TensorBoard。
    - images: [B, 4, H, W]
    - outputs: [B, 3, H, W]
    - masks: [B, 3, H, W]
    """

    def to_3channel_gray(tensor):
        """单通道灰度图变为3通道以便可视化"""
        tensor = tensor.float()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        return tensor.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]

    for i in range(min(images.size(0), max_samples)):
        img = images[i]   # [4, H, W]
        pred = outputs[i] # [3, H, W]
        gt = masks[i]     # [3, H, W]

        # 原图像 4 通道灰度图
        image_gray_list = [to_3channel_gray(img[c]) for c in range(img.size(0))]

        # 模型预测的每通道灰度图（WT, TC, ET）
        pred_gray_list = [to_3channel_gray(pred[c]) for c in range(pred.size(0))]

        # 标签的每通道灰度图
        gt_gray_list = [to_3channel_gray(gt[c]) for c in range(gt.size(0))]

        # 拼图：[图像4通道] + [预测3通道] + [GT 3通道]
        combined = image_gray_list + pred_gray_list + gt_gray_list  # list of [3,H,W]

        grid = make_grid(combined, nrow=4, normalize=False)  # 每行显示4张图

        writer.add_image(f'Output/Sample_{i}', grid, global_step=global_step)
