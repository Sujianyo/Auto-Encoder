import time

import torch
from tqdm import tqdm

from utils.misc import save_and_clear


def forward_pass_without_loss(model, image, device):
    start = time.time()
    outputs = model(image)
    end = time.time()
    time_elapse = end - start

    return outputs, time_elapse


@torch.no_grad()
def inference(model, data_loader, device):
    output_idx = 0

    model.eval()

    tbar = tqdm(data_loader)

    output_file = {'left': [], 'right': [], 'disp_pred': [], 'occ_pred': [], 'time': []}

    for idx, data in enumerate(tbar):
        # forward pass
        outputs, time_elapse = forward_pass_without_loss(model, data, device)

        output_file.append(outputs)

    # save to file
    save_and_clear(output_idx, output_file)

    return