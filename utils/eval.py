from typing import Iterable

import torch
from tqdm import tqdm

from utils.misc import save_and_clear
# from utils.summary_logger import TensorboardSummary


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, device: torch.device,
             epoch: int, save_output: bool):
    model.eval()
    criterion.eval()

    # initialize stats
    eval_stats = {'crs': 0.0}
    # init output file
    if save_output:
        output_idx = 0
        output_file = {'output': []}

    tbar = tqdm(data_loader)
    valid_samples = len(tbar)
    for idx, (image, label, _) in enumerate(tbar):
        # forward pass
        outputs = model(image.to(device))
        losses = criterion(outputs, label.to(device))

        # clear cache
        torch.cuda.empty_cache()

        # save output
        if save_output:
            # save to file
            if len(output_file['left']) >= 50:
                output_idx = save_and_clear(output_idx, output_file)
        eval_stats['crs'] = eval_stats['crs'] + losses

    eval_stats['crs'] = eval_stats['crs']/valid_samples
    # save to file
    if save_output:
        save_and_clear(output_idx, output_file)
    print('Epoch %d, cross entropy %.4f' % 
                (epoch, eval_stats['crs']))
    print()

    return eval_stats