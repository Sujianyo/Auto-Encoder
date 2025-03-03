from model.unet_model import UNet
from model.loss import Criterion
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm

import argparse
import os
import random
import numpy as np
import torch
from utils.checkpoint_saver import Saver
from utils.train import train_one_epoch
from dataset import build_data_loader
from utils.eval import evaluate
from utils.inference import inference
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set unet detector', add_help=False)
    # Data parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') 
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--dataset', default='DRIVE', type=str, help='dataset to train/eval on')
    parser.add_argument('--dataset_directory', default='/mnt/c/dataset', type=str, help='directory to dataset')
    # Training parameters
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=5, type=int, help='Batch size')

    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--ft', action='store_true', help='load model from checkpoint, but discard optimizer state')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='dev', help='checkpoint name for current experiment')
    parser.add_argument('--pre_train', action='store_true')

    # Model parameters
    parser.add_argument('--in_channel', default=3, type=int,
                        help="Size of the embeddings (dimension of the input)")
    parser.add_argument('--out_channel', default=1, type=int,
                        help="Size of the label (dimension of the output)")

    return parser

def save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, best):
    """
    Save current state of training
    """

    # save model
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'best_pred': prev_best
    }

    if best:
        checkpoint_saver.save_checkpoint(checkpoint, 'model.pth.tar', write_best=False)
    else:
        checkpoint_saver.save_checkpoint(checkpoint, 'epoch_' + str(epoch) + '_model.pth.tar', write_best=False)

def print_param(model):
    """
    print number of parameters in the model
    """

    n_parameters = sum(p.numel() for n, p in model.named_parameters())
    print('number of params in backbone:', f'{n_parameters:,}')


def main(args):
    # get device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = UNet(args).to(device)
    print_param(model)


    # define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)


    # initiate saver and logger
    checkpoint_saver = Saver(args)
    # summary_writer = TensorboardSummary(checkpoint_saver.experiment_dir)

    # build dataloader
    data_loader_train, data_loader_test= build_data_loader(args)

    # build loss criterion
    # criterion = build_criterion(args)
    criterion = Criterion()
    prev_best = np.inf
    # Inference 
    if args.inference:
        print("Start inference")
        _, _, data_loader = build_data_loader(args)
        inference(model, data_loader, args.device)

        return

    # train
    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        # train
        print("Epoch: %d" % epoch)
        train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch)


        torch.cuda.empty_cache()
        if args.pre_train or epoch % 10 == 0:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, False)

        eval_stats = evaluate(model, criterion, data_loader_train, device, epoch, False)
        # save if best
        if prev_best > eval_stats['crs']:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, prev_best, checkpoint_saver, True)
    if args.eval:
        print("Start evaluation")
        evaluate(model, criterion, data_loader_val, device, 0, True)
        return

    return
if __name__ == '__main__':
    # Load data
    ap = argparse.ArgumentParser('Unet training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    main(args_)

