# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : Aug 08, 2023
# @File         : backward.py
# @Description  : Class2Simi training

from LibLNL import tools
import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable

class forward_MCL(nn.Module):
    # Forward Meta Classification Likelihood (MCL)

    eps = 1e-12 # Avoid calculating log(0). Use the small value of float16.

    def __init__(self):
        super(forward_MCL, self).__init__()
        return

    def forward(self, prob1, prob2, s_label, q):
        P = prob1.mul(prob2)
        P = P.sum(1)
        P = P * q[0][0] + (1 - P) * q[1][0]
        P.mul_(s_label).add_(s_label.eq(-1).type_as(P))
        negLog_P = -P.add_(forward_MCL.eps).log_()
        return negLog_P.mean()

def class2simi(transition_matrix):
    v00 = v01 = v10 = v11 = 0
    t = transition_matrix
    num_classes = transition_matrix.shape[0]
    for i in range(num_classes):
        for j in range(num_classes):
            a = t[i][j]
            for m in range(num_classes):
                for n in range(num_classes):
                    b = t[m][n]
                    if i == m and j == n:
                        v11 += a * b
                    if i == m and j != n:
                        v10 += a * b
                    if i != m and j == n:
                        v01 += a * b
                    if i != m and j != n:
                        v00 += a * b
    simi_T = np.zeros([2, 2])
    simi_T[0][0] = v11 / (v11 + v10)
    simi_T[0][1] = v10 / (v11 + v10)
    simi_T[1][0] = v01 / (v01 + v00)
    simi_T[1][1] = v00 / (v01 + v00)

    return simi_T

def label2simi(x, mode='cls',mask=None):
    # Convert class label to pairwise similarity
    n=x.nelement()
    assert (n-x.ndimension()+1) == n,'Dimension of Label is not right'
    expand1 = x.view(-1,1).expand(n,n)
    expand2 = x.view(1,-1).expand(n,n)
    out = expand1 - expand2
    out[out!=0] = -1 #dissimilar pair: label=-1
    out[out==0] = 1 #Similar pair: label=1
    if mode=='cls':
        out[out==-1] = 0 #dissimilar pair: label=0
    if mode=='hinge':
        out = out.float() #hingeloss require float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out

def PairEnum(x):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0),1)
    x2 = x.repeat(1,x.size(0)).view(-1,x.size(1))

    return x1,x2

def class2simi_learning(epochs, model, train_loader, val_loader, test_loader, t, optimizer, scheduler):
    val_acc = []
    test_acc = []
    simi_T = class2simi(t)
    criterion = forward_MCL()
    for epoch in range(epochs):
        print('epoch {}'.format(epoch))
        # training-----------------------------
        for i, (inputs, label) in enumerate(train_loader):
            model.train()
            if torch.cuda.is_available():
                model = model.cuda()
                inputs = inputs.cuda()
                label = label.cuda()

            train_target = label2simi(label, mode='hinge').detach()

            out = model(inputs)
            prob = F.softmax(out, dim=1)
            prob1, prob2 = PairEnum(prob)
            loss = criterion(prob1, prob2, train_target, simi_T)
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