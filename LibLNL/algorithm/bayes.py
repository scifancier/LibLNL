# -*- coding: utf-8 -*-
# @Author       : Songhua Wu
# @Time         : Aug 24, 2023
# @File         : bayes.py
# @Description  : Bayes-T Learning

import torch
from torch.nn import functional as F
from LibLNL import tools
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from LibLNL.model import revision_resnet, transition_resnet
from collections import OrderedDict
import torchvision.transforms as transforms
import os
import torch.utils.data as Data
from PIL import Image

def warm_up(model, optimizer, train_loader, test_loader):
    print("Starting warm up.")

    # The optimization loop
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        # training-----------------------------
        for i, (inputs, label, _) in enumerate(train_loader):
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

        print('Warm up epoch {}'.format(epoch))
        accuracy = tools.evaluate(test_loader, model, 'Test')

    return model

def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1).unsqueeze(1)
    T_norm = row_abs / row_sum
    return T_norm

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_forward(model, train_loader, optimizer, Bayesian_T, revision=True):
    train_total = 0
    train_correct = 0

    for i, (data, labels, indexes) in enumerate(train_loader):

        data = data.cuda()
        labels = labels.cuda()
        loss = 0.
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        logits, delta = model(data, revision=True)

        bayes_post = F.softmax(logits, dim=1)

        delta = delta.repeat(len(labels), 1, 1)
        T = Bayesian_T(data)
        if revision == True:
            T = norm(T + delta)
        noisy_post = torch.bmm(bayes_post.unsqueeze(1), T.cuda()).squeeze(1)
        log_noisy_post = torch.log(noisy_post + 1e-12)
        loss = nn.NLLLoss()(log_noisy_post.cuda(), labels.cuda())

        prec1, = accuracy(noisy_post, labels, topk=(1,))
        train_total += 1
        train_correct += prec1
        loss.backward()
        optimizer.step()

    train_acc = float(train_correct) / float(train_total)
    return train_acc

class distilled_dataset(Data.Dataset):
    def __init__(self, distilled_images, distilled_noisy_labels, distilled_bayes_labels, transform=None,
                 target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.distilled_images = distilled_images
        self.distilled_noisy_labels = distilled_noisy_labels
        self.distilled_bayes_labels = distilled_bayes_labels
        # print(self.distilled_images)

    def __getitem__(self, index):
        # print(index)
        img, bayes_label, noisy_label = self.distilled_images[index], self.distilled_bayes_labels[index], \
        self.distilled_noisy_labels[index]
        # print(img)
        # print(bayes_label)
        # print(noisy_label)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            bayes_label, noisy_label = self.target_transform(bayes_label), self.target_transform(noisy_label)

        return img, bayes_label, noisy_label, index

    def __len__(self):
        return len(self.distilled_images)

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


def bayes_learning(train_dataset, train_loader, test_loader, args):

    model_dir = 'bayes'
    if not os.path.exists(model_dir):
        os.system('mkdir -p %s' % model_dir)
    train_loader_batch_1 = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=128,
                                                       num_workers=args.num_workers,
                                                       drop_last=False,
                                                       shuffle=False)


    # Warmup model
    classifier = revision_resnet.ResNet34(10).cuda()
    optimizer_warmup = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    # classifier = warm_up(classifier, optimizer_warmup, train_loader, test_loader)
    # torch.save(classifier.state_dict(), model_dir + '/' + 'warmup_model.pth')

    # Distlled example collection
    threshold = (1 + args.rho) / 2
    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    distilled_example_index_list = []
    distilled_example_labels_list = []
    print('Distilling')
    classifier.eval()
    for i, (data, noisy_label, indexes) in enumerate((train_loader_batch_1)):
        data = data.cuda()
        logits1 = F.softmax(classifier(data), dim=1)
        logits1_max = torch.max(logits1, dim=1)
        mask = logits1_max[0] > threshold
        distilled_example_index_list.extend(indexes[mask])
        distilled_example_labels_list.extend(logits1_max[1].cpu()[mask])
    print("Distilling finished")
    distilled_example_index = np.array(distilled_example_index_list)
    distilled_bayes_labels = np.array(distilled_example_labels_list)
    distilled_images, distilled_noisy_labels = train_dataset.train_data[
        distilled_example_index], train_dataset.train_labels[distilled_example_index]  # noisy labels
    print("Number of distilled examples:" + str(len(distilled_bayes_labels)))

    np.save(model_dir + '/' + 'distilled_images.npy', distilled_images)
    np.save(model_dir + '/' + 'distilled_bayes_labels.npy', distilled_bayes_labels)
    np.save(model_dir + '/' + 'distilled_noisy_labels.npy', distilled_noisy_labels)

    print("Distilled dataset building")

    distilled_images = np.load(model_dir + '/' + 'distilled_images.npy')
    distilled_noisy_labels = np.load(model_dir + '/' + 'distilled_noisy_labels.npy')
    distilled_bayes_labels = np.load(model_dir + '/' + 'distilled_bayes_labels.npy')

    # if args.dataset == 'fmnist':
    #     distilled_dataset_ = data.distilled_dataset(distilled_images,
    #                                                 distilled_noisy_labels,
    #                                                 distilled_bayes_labels,
    #                                                 transform=transforms.Compose([
    #                                                     transforms.ToTensor(),
    #                                                     transforms.Normalize((0.1307,), (0.3081,)), ]),
    #                                                 target_transform=tools.transform_target
    #                                                 )
    if args.dataset == 'cifar10':
        distilled_dataset_ = distilled_dataset(distilled_images,
                                                    distilled_noisy_labels,
                                                    distilled_bayes_labels,
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                             (0.2023, 0.1994, 0.2010)), ]),
                                                    target_transform=transform_target
                                                    )
    # if args.dataset == 'svhn':
    #     distilled_dataset_ = data.distilled_dataset(distilled_images,
    #                                                 distilled_noisy_labels,
    #                                                 distilled_bayes_labels,
    #                                                 transform=transforms.Compose([
    #                                                     transforms.ToTensor(),
    #                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]),
    #                                                 target_transform=tools.transform_target
    #                                                 )

    train_loader_distilled = torch.utils.data.DataLoader(dataset=distilled_dataset_,
                                                         batch_size=args.batch_size,
                                                         num_workers=args.num_workers,
                                                         drop_last=False,
                                                         shuffle=True)

    # if args.dataset == 'fmnist':
    #     Bayesian_T_Network = resnet_transition.ResNet18_F(100)
    #     warm_up_dict = classifier.state_dict()
    #     temp = OrderedDict()
    #     Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
    #     classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    #     for name, parameter in classifier.named_parameters():
    #         if name in Bayesian_T_Network_state_dict:
    #             temp[name] = parameter
    #     Bayesian_T_Network_state_dict.update(temp)
    #     Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)
    # if args.dataset == 'svhn':
    #     Bayesian_T_Network = resnet_transition.ResNet34(100)
    #     warm_up_dict = classifier.state_dict()
    #     temp = OrderedDict()
    #     Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
    #     classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    #     for name, parameter in classifier.named_parameters():
    #         if name in Bayesian_T_Network_state_dict:
    #             temp[name] = parameter
    #     Bayesian_T_Network_state_dict.update(temp)
    #     Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)
    if args.dataset == 'cifar10':
        Bayesian_T_Network = transition_resnet.ResNet34(100)
        temp = OrderedDict()
        Bayesian_T_Network_state_dict = Bayesian_T_Network.state_dict()
        classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
        for name, parameter in classifier.named_parameters():
            if name in Bayesian_T_Network_state_dict:
                temp[name] = parameter
        Bayesian_T_Network_state_dict.update(temp)
        Bayesian_T_Network.load_state_dict(Bayesian_T_Network_state_dict)
    Bayesian_T_Network.cuda()
    # Learning Bayes T
    #     clf_bayes_output -> transition matrix with size c*c
    optimizer_bayes = torch.optim.SGD(Bayesian_T_Network.parameters(), lr=0.01, momentum=0.9)
    loss_function = nn.NLLLoss()
    for epoch in range(0, 50):
        bayes_loss = 0.
        Bayesian_T_Network.train()
        for data, bayes_labels, noisy_labels, index in train_loader_distilled:
            data = data.cuda()
            bayes_labels, noisy_labels = bayes_labels.cuda(), noisy_labels.cuda()
            batch_matrix = Bayesian_T_Network(data)  # batch_size x 10 x 10
            noisy_class_post = torch.zeros((batch_matrix.shape[0], 10))
            for j in range(batch_matrix.shape[0]):
                bayes_label_one_hot = torch.nn.functional.one_hot(bayes_labels[j], 10).float()  # 1*10
                bayes_label_one_hot = bayes_label_one_hot.unsqueeze(0)
                noisy_class_post_temp = bayes_label_one_hot.float().mm(batch_matrix[j])  # 1*10 noisy
                noisy_class_post[j, :] = noisy_class_post_temp
        noisy_class_post = torch.log(noisy_class_post + 1e-12)
        loss = loss_function(noisy_class_post.cuda(), noisy_labels)
        optimizer_bayes.zero_grad()
        loss.backward()
        optimizer_bayes.step()
        bayes_loss += loss.item()
        print('Training Epoch [%d], Loss: %.4f' % (epoch + 1, loss.item()))
        torch.save(Bayesian_T_Network.state_dict(), model_dir + '/' + 'BayesianT.pth')


    # loss_correction
    val_acc_list = []
    test_acc_list = []

    classifier.load_state_dict(torch.load(model_dir + '/' + 'warmup_model.pth'))
    nn.init.constant_(classifier.T_revision.weight, 0.0)
    Bayesian_T_Network.load_state_dict(torch.load(model_dir + '/' + 'BayesianT.pth'))
    optimizer_r = torch.optim.Adam(classifier.parameters(), lr=5e-7, weight_decay=1e-4)

    for epoch in range(0, args.epochs):
        print('Epoch {}'.format(epoch))

        classifier.train()
        Bayesian_T_Network.eval()
        train_total = 0
        train_correct = 0
        train_acc = train_forward(classifier, train_loader, optimizer_r, Bayesian_T_Network, revision=True)
        test_acc = tools.evaluate(test_loader, classifier, 'Test')
        test_acc_list.append(test_acc)

    id = np.argmax(np.array(test_acc_list))
    test_acc_max = test_acc_list[id]
    print('Test Acc: ')
    print(test_acc_max)
    return test_acc_max