import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
import torch
from torch.utils.data.dataset import random_split

def transform_data(dataset, M= 0, train_batch_size= 128, test_batch_size= 10, validation= False, down= False):
    trainset, valset, testset = load_data(dataset)

    if down:
        trainset = downsample(trainset, M)
        #testset = downsample(testset, M)

    num_classes = 10

    # create data loaders
    trainloader = DataLoader(trainset, batch_size= train_batch_size, shuffle=True)
    if valset is not None:
        validation_loader = DataLoader(trainset, batch_size= test_batch_size, shuffle=True)
    else:
        validation_loader = None
    testloader = DataLoader(testset, batch_size= test_batch_size, shuffle=True)

    return trainloader, testloader, validation_loader, num_classes


def load_data(dataset):
    if dataset.lower() == 'cifar10':
        # convert each image to tensor format
        transform_train = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # source https://github.com/kuangliu/pytorch-cifar/issues/19
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            # source https://github.com/kuangliu/pytorch-cifar/issues/19
        ])

        trainset = CIFAR10("./data", train=True, download=True, transform=transform_train)
        testset = CIFAR10("./data", train=False, download=True, transform=transform_test)

        return trainset, None, testset

    elif dataset.lower() == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])

        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        valset = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

        return trainset, valset, testset
    
    else:
        raise ValueError('Dataset not specified.')


def downsample(dataset, M):
    labels = dataset.class_to_idx
    label_counts = {key:0 for key in labels.values()}
    samples_index = []

    for inx, item in enumerate(dataset):

        if all(count >= M for count in label_counts.values()):
            break
        else:
            data_item, label = item
            if label_counts[label] < M:
                label_counts[label] += 1
                samples_index.append(inx)

    data_subset = Subset(dataset, samples_index)
    return data_subset
