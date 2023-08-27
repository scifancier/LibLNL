# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : Aug 10, 2023
# @File         : robust_loss_example.py
# @Description  : Example to run robust_loss methods with LibLNL

import argparse
import os
import sys
import numpy as np
import torch
from LibLNL.data.data_loader import load_data
from LibLNL.model import resnet
from LibLNL.algorithm.estimate_t_anchor import estimate_t_anchor
from LibLNL.algorithm.forward import forward_learning
from LibLNL.algorithm.bayes import bayes_learning
from LibLNL import tools
from LibLNL.algorithm.robust_loss import *

from torch.nn import functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--noise_rate', type=float, default=0.2, help="noise rate")
parser.add_argument('--noise_type', type=str, default='symmetric', help="noise type: symmetric or instance-dependent")
parser.add_argument('--print', type=int, default=1, help="1 for printing in terminal, 0 for printing in file")
parser.add_argument('--dataset', type=str, default='cifar10', help="mnist, cifar10")
parser.add_argument('--data_percent', default=0.9, type=float, help='data number percent')
parser.add_argument('--num_class', default=10, type=int, help="number of classes")
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, help='weight_decay for training', default=0)
parser.add_argument('--epochs', default=60, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--num_workers', type=int, default=16, help="thread for dataloader")
parser.add_argument('--gpu_id', type=str, default='1')
parser.add_argument('--index', type=bool, default=False, help='whether need index in dataloader')
parser.add_argument('--loss', type=str, default='SCE')

args = parser.parse_args()


def main():

    print('\nLoading data ...')

    ''' 
        artificially add noise of args.noise_type with args.noise_type to the data,
        if data is noisy originally, please set args.noise_type=none  
    '''



    train_dataset, train_loader, val_loader, est_loader, test_loader, train_data_num, true_t = load_data(args)

    if args.net == 'resnet34':
        model = resnet.ResNet34(args.num_class)

    # Robust loss training
    optim_args = {'lr': args.lr, 'weight_decay': args.weight_decay}
    optimizer = torch.optim.__dict__[args.optimizer](model.parameters(), **optim_args)
    scheduler = torch.optim.lr_scheduler.__dict__[args.scheduler](optimizer, step_size=40, gamma=1 / 3)

    loss_classes = {
        'SCE': SCELoss,
        'RCE': ReverseCrossEntropy,
        'NRCE': NormalizedReverseCrossEntropy,
        'NCE': NormalizedCrossEntropy,
        'GCE': GeneralizedCrossEntropy,
        'NGCE': NormalizedGeneralizedCrossEntropy,
        'MAE': MeanAbsoluteError,
        'NMAE': NormalizedMeanAbsoluteError,
        'NCEandRCE': NCEandRCE,
        'NCEandMAE': NCEandMAE,
        'GCEandMAE': GCEandMAE,
        'GCEandRCE': GCEandRCE,
        'GCEandNCE': GCEandNCE,
        'NGCEandNCE': NGCEandNCE,
        'NGCEandMAE': NGCEandMAE,
        'NGCEandRCE': NGCEandRCE,
        'MAEandRCE': MAEandRCE,
        'NLNL': NLNL,
        'FL': FocalLoss,
        'NFL': NormalizedFocalLoss,
        'NFLandNCE': NFLandNCE,
        'NFLandMAE': NFLandMAE,
        'NFLandRCE': NFLandRCE,
        'DMI': DMILoss
    }

    robust_loss = loss_classes[args.loss]()

    robust_loss_learning(robust_loss, args.epochs, model, train_loader, val_loader, test_loader, optimizer, scheduler)


if __name__ == "__main__":

    args.net = 'resnet34'
    args.optimizer = 'Adam'
    args.scheduler = 'StepLR'

    tools.set_seed(args.seed)

    if args.print == 0:
        f = open(args.dataset + '_0.4.txt', 'a')
        sys.stdout = f
        sys.stderr = f

    print('Parameter:', args)

    if args.gpu_id != 'n':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print(os.environ["CUDA_VISIBLE_DEVICES"])

    main()

