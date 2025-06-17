import torch.utils.data as data

from dataset.drive import DRIVE_Dataset
from dataset.brain_35h import Br35H
# from dataset.brain_lgg import 

def build_data_loader(args):
    '''
    Build data loader

    :param args: arg parser object
    :return: train, test dataloaders
    '''
    if args.dataset_directory == '':
        raise ValueError(f'Dataset directory cannot be empty.')
    else:
        dataset_dir = args.dataset_directory
    if args.dataset == 'DRIVE' or args.dataset == 'drive':
        dataset_train = DRIVE_Dataset(dataset_dir, 'train')
        dataset_val = []
        dataset_test = DRIVE_Dataset(dataset_dir, 'test')
    elif args.dataset == 'lgg_brain':
        dataset_train
        dataset_test
    elif args.dataset == 'br35h':
        dataset_train = Br35H(dataset_dir, 'train')
        dataset_val = Br35H(dataset_dir, 'val')
        dataset_test = Br35H(dataset_dir, 'test')
    data_loader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    data_loader_val = data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    data_loader_test = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    return data_loader_train, data_loader_val, data_loader_test