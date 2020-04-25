import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def transform_data():
    # convert each image to tensor format
    transform = transforms.Compose([
        transforms.ToTensor()  # convert to tensor
    ])

    # load data
    trainset = MNIST(".", train=True, download=True, transform=transform)
    testset = MNIST(".", train=False, download=True, transform=transform)

    # create data loaders
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=True)


    return trainloader, testloader