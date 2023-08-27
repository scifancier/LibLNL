# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : Aug 10, 2023
# @File         : mixup.py
# @Description  : Mixup training

import torch
from torch.nn import functional as F
from LibLNL import tools
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_learning(epochs, model, train_loader, val_loader, test_loader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    val_acc = []
    test_acc = []
    for epoch in range(epochs):
        print('epoch {}'.format(epoch))
        # training-----------------------------
        for i, (inputs, label) in enumerate(train_loader):
            model.train()
            if torch.cuda.is_available():
                model = model.cuda()
                inputs = inputs.cuda()
                label = label.cuda()

            inputs, targets_a, targets_b, lam = mixup_data(inputs, label, use_cuda=True)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_acc.append(tools.evaluate(val_loader, model, 'Validation'))
        test_acc.append(tools.evaluate(test_loader, model, 'Test'))

    index = np.argmax(np.array(val_acc))
    test_acc = test_acc[index]
    print('Test Acc: ')
    print(test_acc)

