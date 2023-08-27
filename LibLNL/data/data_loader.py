import torchvision
from LibLNL.data.data_transform import get_transform
import torchvision.transforms as transforms
from LibLNL.data.NoisyUtil import Train_Dataset, Train_Dataset_Index, dataset_split, Semi_Unlabeled_Dataset
import numpy as np
from torch.utils.data import DataLoader
import torch



data_info_dict = {
    "CIFAR10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "root": "~/.torchvision/datasets/cifar10",
        'random_crop': 32
    },
    "CIFAR100": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "root": "~/.torchvision/datasets/cifar100",
        'random_crop': 32
    },
    "SVHN": {
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "root": "~/.torchvision/datasets/SVHN",
        'random_crop': 32
    },
    "MNIST": {
        "mean": (0.1306604762738429,),
        "std": (0.30810780717887876,),
        "root": "~/.torchvision/datasets/MNIST",
        'random_crop': 28
    },
    "FASHIONMNIST": {
        "mean": (0.286,),
        "std": (0.353,),
        "root": "~/.torchvision/datasets/FashionMNIST",
        'random_crop': 28
    }
}

def get_transform(dataset):
    info = data_info_dict[dataset]

    transform_train = transforms.Compose([
        transforms.RandomCrop(info["random_crop"], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(info["mean"], info["std"]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(info["mean"], info["std"]),
    ])

    return transform_train, transform_test


def load_data(args):
    dataset = args.dataset.upper()
    transform_train, transform_test = get_transform(dataset)
    info = data_info_dict[dataset]
    root = info["root"]
    train_set = torchvision.datasets.__dict__[dataset](root=root, train=True, download=True)
    test_set = torchvision.datasets.__dict__[dataset](root=root, train=False, transform=transform_test, download=True)

    train_data, val_data, train_noisy_labels, val_noisy_labels, train_clean_labels, _, transition_matrix = dataset_split(train_set.data,
                                                                                                      np.array(
                                                                                                          train_set.targets),
                                                                                                      args.noise_rate,
                                                                                                      args.noise_type,
                                                                                                      args.data_percent,
                                                                                                      args.seed,
                                                                                                      args.num_class)
    if args.index:
        train_dataset = Train_Dataset_Index(train_data, train_noisy_labels, transform_train)
        val_dataset = Train_Dataset_Index(val_data, val_noisy_labels, transform_train)
    else:
        train_dataset = Train_Dataset(train_data, train_noisy_labels, transform_train)
        val_dataset = Train_Dataset(val_data, val_noisy_labels, transform_train)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True)
    est_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=8,
                            pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=8,
                             pin_memory=True)

    return train_dataset, train_loader, val_loader, est_loader, test_loader, len(train_dataset), torch.from_numpy(transition_matrix).float().cuda()
