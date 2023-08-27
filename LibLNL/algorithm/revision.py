# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : Aug 09, 2023
# @File         : revision.py
# @Description  : T-revision learning

import torch
from torch.nn import functional as F
from LibLNL import tools
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class reweight_loss(nn.Module):
    def __init__(self):
        super(reweight_loss, self).__init__()

    def forward(self, out, T, target):
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
            out_T = torch.matmul(T.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]
            beta = pro1 / pro2
            beta = Variable(beta, requires_grad=True)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)


class reweighting_revision_loss(nn.Module):
    def __init__(self):
        super(reweighting_revision_loss, self).__init__()

    def forward(self, out, T, correction, target):
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
            T = T + correction
            T_result = T
            out_T = torch.matmul(T_result.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]
            beta = (pro1 / pro2)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)


def get_noisy_prob(transition_mat, clean_prob):
    return torch.matmul(transition_mat.T, clean_prob.unsqueeze(-1)).squeeze()


class reweight_loss_v2(nn.Module):
    def __init__(self):
        super(reweight_loss_v2, self).__init__()

    def forward(self, out, T, target):
        out_softmax = F.softmax(out, dim=1)
        noisy_prob = get_noisy_prob(T, out_softmax)
        pro1 = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()
        pro2 = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()
        beta = pro1 / pro2
        beta = Variable(beta, requires_grad=True)
        cross_loss = F.cross_entropy(out, target, reduction='none')
        _loss = beta * cross_loss
        return torch.mean(_loss)


class reweighting_revision_loss_v2(nn.Module):
    def __init__(self):
        super(reweighting_revision_loss_v2, self).__init__()

    def forward(self, out, T, correction, target):
        out_softmax = F.softmax(out, dim=1)
        T = T + correction
        noisy_prob = get_noisy_prob(T, out_softmax)
        pro1 = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()
        pro2 = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()
        beta = pro1 / pro2
        beta = Variable(beta, requires_grad=True)
        cross_loss = F.cross_entropy(out, target, reduction='none')
        _loss = beta * cross_loss
        return torch.mean(_loss)

def revision_learning(epoch_revision, epochs, model, train_loader, val_loader, test_loader, T, optimizer, scheduler):
    val_acc_list_r = []
    # loss
    loss_func_ce = nn.CrossEntropyLoss()
    loss_func_reweight = reweight_loss()
    loss_func_revision = reweighting_revision_loss()
    optimizer_revision = torch.optim.Adam(model.parameters(), lr=5e-7, weight_decay=1e-4)
    # cuda
    if torch.cuda.is_available:
        model = model.cuda()
        loss_func_ce = loss_func_ce.cuda()
        loss_func_reweight = loss_func_reweight.cuda()
        loss_func_revision = loss_func_revision.cuda()

    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.
        best_val = 0.
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            out = model(batch_x, revision=False)
            prob = F.softmax(out, dim=1)
            prob = prob.t()
            loss = loss_func_reweight(out, T, batch_y)
            out_forward = torch.matmul(T.t(), prob)
            out_forward = out_forward.t()
            train_loss += loss.item()
            pred = torch.max(out_forward, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = tools.evaluate(val_loader, model, 'Validation')
        tools.evaluate(test_loader, model, 'Test')
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), 'best.pth')

    reweight_model_path = 'best.pth'
    reweight_model_path = torch.load(reweight_model_path)
    model.load_state_dict(reweight_model_path)
    nn.init.constant_(model.T_revision.weight, 0.0)

    print('Revision......')

    for epoch in range(epoch_revision):

        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer_revision.zero_grad()
            out, correction = model(batch_x, revision=True)
            prob = F.softmax(out, dim=1)
            prob = prob.t()
            loss = loss_func_revision(out, T, correction, batch_y)
            out_forward = torch.matmul((T + correction).t(), prob)
            out_forward = out_forward.t()
            train_loss += loss.item()
            pred = torch.max(out_forward, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            loss.backward()
            optimizer_revision.step()

        tools.evaluate(test_loader, model, 'Test')
        tools.evaluate(val_loader, model, 'Validation')

