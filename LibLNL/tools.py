# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : 8/7/23 9:40 pm
# @File         : tools.py
# @Description  :

import torch
import numpy as np
import os


def evaluate(eval_loader, model, s):
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    model.eval()
    for i, (inputs, target) in enumerate(eval_loader):

        if torch.cuda.is_available():
            with torch.no_grad():
                model = model.cuda()
                inputs = inputs.cuda()
                target = target.cuda()

        output = model(inputs)
        prediction = torch.argmax(output, 1)
        correct += (prediction == target).sum().float()
        total += len(target)
        output = output.detach()

    acc = (correct / total).detach()

    print(s + ' ACC: % .4f' % acc.item())

    return acc.item()


def set_seed(seed):
    # set seed
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
