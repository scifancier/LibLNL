# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : 8/7/23 10:22 pm
# @File         : forward.py
# @Description  : Forward training

import torch
from torch.nn import functional as F
from LibLNL import tools
import numpy as np


def forward_learning(epochs, model, train_loader, val_loader, test_loader, t, optimizer, scheduler):
    nllloss = torch.nn.NLLLoss()
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
            prob = F.softmax(out, dim=1)
            prob_t = prob.mm(t)
            logprob = prob_t.log()
            loss = nllloss(logprob, label)
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
