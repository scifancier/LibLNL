import torchvision.transforms as transforms

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
    dataset = dataset.upper()
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
