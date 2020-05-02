import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN


def transform_data(M):

    transform_train, transform_test = transform('cifar10')

    # load data
    trainset = CIFAR10(".", train=True, download=True, transform=transform_train)
    testset = CIFAR10(".", train=False, download=True, transform=transform_test)

    trainset = downsample(trainset, M)
    #testset = downsample(testset, M)


    # create data loaders

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=10, shuffle=True)



    return trainloader, testloader


def transform(dataset):

    if dataset == 'cifar10':
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

        return transform_train, transform_test


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









def get_loaders(dataset, M):

    if dataset.lower() == 'cifar10':
        trainloader, testloader = transform_data(M)
        return trainloader, testloader



if __name__ == '__main__':


    get_loaders('cifar10', 10)
