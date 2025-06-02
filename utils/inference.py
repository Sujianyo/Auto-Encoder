import time

import torch
from tqdm import tqdm



def forward_pass_without_loss(model, x, device):
    start = time.time()
    outputs = model(x)
    end = time.time()
    time_elapse = end - start

    return outputs, time_elapse


@torch.no_grad()
def inference(model, data_loader, device):
    output_idx = 0

    model.eval()

    tbar = tqdm(data_loader)

    output_file = {'image': [], 'pre': [], 'time': []}

    for idx, x in enumerate(tbar):
        # forward pass
        outputs, time_elapse = forward_pass_without_loss(model, x, device)

        output_file.append(outputs)

    return