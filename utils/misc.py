import torch
def save_and_clear(idx, output_file):
    with open('output-' + str(idx) + '.dat', 'wb') as f:
        torch.save(output_file, f)
    idx += 1

    # clear
    for key in output_file:
        output_file[key].clear()

    return idx