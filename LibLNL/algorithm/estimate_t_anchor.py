# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : 8/7/23 9:21 pm
# @File         : estimate_t_anchor.py
# @Description  : Estimate transition matrix with anchor point.

import torch
from LibLNL import tools
from torch.nn import functional as F
import numpy as np

def fit(x, num_classes, per_radio=97):

    c = num_classes
    t = np.empty((c, c))
    eta_corr = x
    for i in np.arange(c):
        eta_thresh = np.percentile(eta_corr[:, i], per_radio, interpolation='higher')
        robust_eta = eta_corr[:, i]
        robust_eta[robust_eta >= eta_thresh] = 0.0
        idx_best = np.argmax(robust_eta)
        for j in np.arange(c):
            t[i, j] = eta_corr[idx_best, j]
    return t


def estimate_t_anchor(epochs, train_loader, est_loader, val_loader, batch_size, train_data_num, num_class, model, optimizer):

    print("Estimating transition matrix with anchor point.")
    # prob save files
    prob_matrix = torch.zeros((train_data_num, num_class))

    # The optimization loop
    criterion = torch.nn.CrossEntropyLoss()
    best_accuracy = 0

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
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = tools.evaluate(val_loader, model, 'Validation')

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            for i, (inputs, label) in enumerate(est_loader):
                model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()
                    inputs = inputs.cuda()

                out = model(inputs)
                prob = F.softmax(out, dim=1)
                prob = prob.detach()

                # save probability matrix
                prob_matrix[i * batch_size:(i + 1) * batch_size, :] = prob

    transition_matrix = fit(prob_matrix, num_class, per_radio=100)
    print("\nTransition matrix: \n", transition_matrix)

    return torch.from_numpy(transition_matrix).float().cuda()
