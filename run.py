from simCLR import SimCLR
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='BraTS', help='model name: (default: arch+timestamp')
    parser.add_argument('--dataset', default="Brats2020TrainDataset",
                        help='dataset name')
    parser.add_argument('--input_channel', default=4, type=int, help='input channels')
    parser.add_argument('--output_channel', default=3, type=int, help='input channels')
    parser.add_argument('--image-ext', default='npy', help='image file extension')
    parser.add_argument('--mask-ext', default='npy', help='mask file extension')
    parser.add_argument('--loss', default='BCEDiceLoss')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=32, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False,
                        help='nesterov')
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--validate_frequency', default=1, type=int)
    parser.add_argument('--print_interval', default=10, type=int)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    torch.cuda.manual_seed_all(1)

    simclr = SimCLR(args)
    simclr.train()

if __name__ == "__main__":
    main()