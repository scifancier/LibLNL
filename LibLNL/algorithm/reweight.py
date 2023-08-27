# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : Aug 04, 2023
# @File         : reweight.py
# @Description  : Reweighting training

import torch
from torch.nn import functional as F
from LibLNL import tools
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def reweight_loss(out, t, target):
    loss = 0.
    out_softmax = F.softmax(out, dim=1)
    for i in range(len(target)):
        temp_softmax = out_softmax[i]
        temp = out[i]
        temp = torch.unsqueeze(temp, 0)
        temp_softmax = torch.unsqueeze(temp_softmax, 0)
        temp_target = target[i]
        temp_target = torch.unsqueeze(temp_target, 0)
        pro1 = temp_softmax[:, target[i]]
        out_T = torch.matmul(t.t(), temp_softmax.t())
        out_T = out_T.t()
        pro2 = out_T[:, target[i]]
        beta = pro1 / pro2
        beta = Variable(beta, requires_grad=True)
        cross_loss = F.cross_entropy(temp, temp_target)
        _loss = beta * cross_loss
        loss += _loss
    return loss / len(target)

def reweight_learning(epochs, model, train_loader, val_loader, test_loader, t, optimizer, scheduler):
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

            out = model(inputs)
            loss = reweight_loss(out, t, label)
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

