import argparse

def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--experiment_name', default='lgg_brain', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--patch_size', default=7, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--ft', action='store_true', help='load model from checkpoint, but discard optimizer state')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--checkpoint', type=str, default='dev', help='checkpoint name for current experiment')
    parser.add_argument('--pre_train', action='store_true')
    # parser.add_argument('--downsample', default=3, type=int,
    #                     help='Default downsampling is 4 and cannot be changed.')
    # parser.add_argument('--apex', action='store_true', help='enable mixed precision training')

    # parser.add_argument('--channel_dim', default=128, type=int,
    #                     help="Size of the embeddings (dimension of the transformer)")
    # Transformer
    parser.add_argument('--patchify', action='store_true', help='Whether to use patch and then attention')
    parser.add_argument('--axial', action='store_true', help='Whether to use Axial attention')
    parser.add_argument('--num_layer', default=4, type=int)

    # * Dataset parameters
    parser.add_argument('--dataset', default='lgg_brain', type=str, help='dataset to train/eval on')
    parser.add_argument('--dataset_directory', default='', type=str, help='directory to dataset')
    parser.add_argument('--validation', default='validation', type=str, choices={'validation', 'validation_all'},
                        help='If we validate on all provided training images')

    return parser