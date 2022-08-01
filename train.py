import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from dataset import BratsSupervisedDataset, BratsValidationDataset

from loss.BCEDiceLoss import BCEDiceLoss
from utils import count_params
import pandas as pd

# from torchmetrics import Dice
from metrics import compute_dice
import albumentations as A
from unet import Unet

import pickle
'''
Supervised Training and Validating on the source domain
'''
device = torch.device('cuda:0')
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='BraTS', help='model name: (default: arch+timestamp')
    parser.add_argument('--dataset', default="Brats2020TrainDataset",
                        help='dataset name')
    parser.add_argument('--input_channel', default=4, type=int, help='input channels')
    parser.add_argument('--output_channel', default=3, type=int, help='input channels')
    parser.add_argument('--image-ext', default='npy', help='image file extension')
    parser.add_argument('--mask-ext', default='npy', help='mask file extension')
    parser.add_argument('--aug', default=True)
    parser.add_argument('--loss', default='BCEDiceLoss')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=30, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False,
                        help='nesterov')
    parser.add_argument('--evaluate_frequency', default=10, type=int)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--temperature', default=0.5, type=float)
    args = parser.parse_args()

    return args

def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    # dice = Dice(num_classes=3, average=None)
    threshold = 0.5
    model.train()
    loss_ = []
    dice_ = []
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.use_cuda:
            input = input.to(device)
            target = target.to(device)

        output = model(input)
        loss = criterion(output, target)
        # Dice = dice((torch.sigmoid(output).cpu().detach() > 0.5).int().numpy(), target.cpu().detach().numpy())
        Dice = compute_dice(output, target)
        loss_.append(loss.item())
        dice_.append(Dice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_avg = np.mean(loss_)
    dice_avg = np.mean(dice_)
    log = OrderedDict([
        ('loss', loss_avg),
        ('dice', dice_avg)
    ])
    return log

def validate(args, val_loader, model, criterion):
    # dice = Dice(num_classes=3, average=None)
    threshold = 0.5
    # switch to evaluate mode
    model.eval()
    loss_ = []
    dice_ = []
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if args.use_cuda:
                input = input.to(device)
                target = target.to(device)
            with torch.no_grad():
                output = model(input)
            loss = criterion(output, target)
            # Dice = dice((torch.sigmoid(output).cpu().detach() > 0.5).int().numpy(), target.cpu().detach().numpy())
            Dice = compute_dice(output, target)
            loss_.append(loss.item())
            dice_.append(Dice)

    loss_avg = np.mean(loss_)
    dice_avg = np.mean(dice_)

    log = OrderedDict([
        ('loss', loss_avg),
        ('dice', dice_avg),
    ])

    return log

def main():
    torch.cuda.manual_seed_all(1)

    args = parse_args()
    #args.dataset = "datasets"
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    criterion = BCEDiceLoss()

    cudnn.benchmark = True

    # Data loading code
    img_paths = glob('/mnt/asgard2/data/lingkai/braTS20/slice/BraTS17_TCIA_HGG/image/*')
    mask_paths = glob('/mnt/asgard2/data/lingkai/braTS20/slice/BraTS17_TCIA_HGG/label/*')
    # val_img_paths = glob('/mnt/asgard2/data/lingkai/braTS20/slice/ImageTs/*')
    # val_mask_paths = glob('/mnt/asgard2/data/lingkai/braTS20/slice/LabelTs/*')
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))
    with open("trainImgPath", "wb") as fp:
        pickle.dump(train_img_paths, fp)
        
    with open("trainLabelPath", "wb") as fp:
        pickle.dump(train_mask_paths, fp)

    with open("valImgPath", "wb") as fp:
        pickle.dump(val_img_paths, fp)
        
    with open("valLabelPath", "wb") as fp:
        pickle.dump(val_mask_paths, fp)

    # create model
    model = Unet(in_channel=args.input_channel, out_channel=args.output_channel)
    if args.use_cuda:
        model = model.to(device)

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    # data augmentation
    train_transform = A.Compose(
        [
        # A.CropNonEmptyMaskIfExists(height=150, width=150),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(1.0, 1.5), p=0.15),
        A.Affine(translate_percent=(0, 0.25), p=0.5),
        A.GaussianBlur(sigma_limit=(0.5, 1.5), p=0.15), 
        A.GaussNoise(var_limit=(0, 0.33), p=0.15),
        A.RandomBrightness(limit=(0.7, 1.3), p=0.15),
        A.RandomContrast(limit=(0.65, 1.5), p=0.15)],
        additional_targets={'t1': 'image', 't1ce': 'image', 't2': 'image', 'tumorCore': 'mask', 'enhancingTumor': 'mask'}
    )

    train_dataset = BratsSupervisedDataset(train_img_paths, train_mask_paths, augmentation=train_transform)
    val_dataset = BratsValidationDataset(val_img_paths, val_mask_paths)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'dice', 'val_loss', 'val_dice'
    ])

    best_dice = 0
    trigger = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        print('loss: %.4f - dice: %.4f - val_loss: %.4f - val_dice: %.4f'
            %(train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))

        tmp = pd.DataFrame([[epoch,
            args.lr,
            train_log['loss'],
            train_log['dice'],
            val_log['loss'],
            val_log['dice'],
        ]], columns=['epoch', 'lr', 'loss', 'dice', 'val_loss', 'val_dice'])

        # log = log.append(tmp, ignore_index=True)
        log = pd.concat([log, tmp])
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
    



