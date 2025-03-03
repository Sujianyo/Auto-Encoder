import torch.utils.data as data

from dataset.drive import DRIVE_Dataset



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
    if args.dataset == 'DRIVE' or 'drive':
        dataset_train = DRIVE_Dataset(dataset_dir, 'train')
        dataset_test = DRIVE_Dataset(dataset_dir, 'test')

    data_loader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    data_loader_test = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    return data_loader_train, data_loader_test