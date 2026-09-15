"""task.py adattato per Python 3.6 e Jetson Nano con FER-2013."""

import os
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, ToTensor, Normalize


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(partition_id: int, num_partitions: int, batch_size: int = 16):
    """Carica FER-2013 da cartella locale con ImageFolder e imposta i DataLoader per risparmiare RAM."""
    transform = Compose([
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    trainset = ImageFolder(root="/home/picocluster/flower_node/data/train", transform=transform)
    testset = ImageFolder(root="/home/picocluster/flower_node/data/test", transform=transform)

    num_samples_train = len(trainset) // num_partitions
    train_indices = list(range(partition_id * num_samples_train, (partition_id + 1) * num_samples_train))
    
    num_samples_test = len(testset) // num_partitions
    test_indices = list(range(partition_id * num_samples_test, (partition_id + 1) * num_samples_test))

    # num_workers=0 evita l'uso di multiprocessing che satura la RAM della Jetson Nano
    trainloader = DataLoader(
        Subset(trainset, train_indices), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    testloader = DataLoader(
        Subset(testset, test_indices), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )

    return trainloader, testloader


def load_centralized_dataset(batch_size: int = 64):
    """Carica il test set globale sul server."""
    transform = Compose([
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    testset = ImageFolder(root="/app/data/test", transform=transform)
    return DataLoader(testset, batch_size=batch_size, shuffle=False)


def train(net, trainloader, epochs: int, lr: float, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / (epochs * len(trainloader)) if len(trainloader) > 0 else 0.0


def test(net, testloader, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset) if len(testloader.dataset) > 0 else 0.0
    loss = loss / len(testloader) if len(testloader) > 0 else 0.0
    return loss, accuracy