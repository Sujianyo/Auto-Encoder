from typing import Iterable

import torch
from tqdm import tqdm




def iou_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Intersection over Union for a tensor of predictions."""
    predictions = (predictions > 0.5).float()  # Convert to binary values (0 or 1)
    labels = labels.float()

    # Ensure tensors have shape (batch, H, W)
    if predictions.dim() == 4:  # Shape (batch, 1, H, W)
        predictions = predictions.squeeze(1)
    if labels.dim() == 4:
        labels = labels.squeeze(1)

    intersection = (predictions * labels).sum((1, 2))  # Element-wise multiplication
    union = (predictions + labels - predictions*labels).clamp(0, 1).sum((1, 2))  # Sum without overcounting

    return (intersection + e) / (union + e)  # Add epsilon for numerical stability

def dice_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Dice coefficient for a tensor of predictions."""
    predictions = (predictions > 0.5).float()
    labels = labels.float()

    if predictions.dim() == 4:
        predictions = predictions.squeeze(1)
    if labels.dim() == 4:
        labels = labels.squeeze(1)

    intersection = (predictions * labels).sum((1, 2))
    return (2 * intersection + e) / (predictions.sum((1, 2)) + labels.sum((1, 2)) + e)

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, device: torch.device,
             epoch: int, save_output: bool):
    model.eval()
    criterion.eval()

    # initialize stats
    eval_stats = {'crs': 0.0, 'dice': 0.0, 'iou': 0.0}
    # init output file
    if save_output:
        output_idx = 0
        output_file = {'output': []}

    tbar = tqdm(data_loader)
    valid_samples = len(tbar)
    for idx, (image, label) in enumerate(tbar):
        # forward pass
        outputs = model(image.to(device))
        losses = criterion(outputs, label.to(device))
        dice = dice_pytorch(outputs, label.to(device)).mean()
        iou = iou_pytorch(outputs, label.to(device)).mean()
        # clear cache
        torch.cuda.empty_cache()
        eval_stats['crs'] = eval_stats['crs'] + losses
        eval_stats['iou'] = eval_stats['iou'] + iou
        eval_stats['dice'] = eval_stats['dice'] + dice
    eval_stats['crs'] = eval_stats['crs']/valid_samples
    eval_stats['iou'] = eval_stats['iou']/valid_samples
    eval_stats['dice'] = eval_stats['dice']/valid_samples
    # save to file

    print('Epoch %d, cross entropy %.4f, dice %.4f, iou %.4f' % 
                (epoch, eval_stats['crs'], eval_stats['dice'], eval_stats['iou']))
    print()

    return eval_stats