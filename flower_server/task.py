"""task.py adattato per Python 3.6 e Jetson Nano con FER-2013."""

import os
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, ToTensor, Normalize

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def _iid_split(num_samples: int, num_partitions: int, partition_id: int, seed: int = SEED):
    """Shuffle con seed fisso, poi split in blocchi contigui."""
    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    per_partition = num_samples // num_partitions
    start = partition_id * per_partition
    end = start + per_partition
    return indices[start:end].tolist()

def _dirichlet_split(targets, num_partitions: int, partition_id: int, alpha: float = 0.5, seed: int = SEED):
    """Partizionamento non-IID per classe dominante, via distribuzione di Dirichlet.
    alpha basso -> ogni client vede quasi solo 1-2 classi (molto sbilanciato).
    alpha alto  -> distribuzione quasi uniforme tra client (vicino a IID)."""
    rng = np.random.RandomState(seed)
    targets = np.array(targets)
    num_classes = len(np.unique(targets))

    partitions = [[] for _ in range(num_partitions)]

    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(alpha * np.ones(num_partitions))
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        class_splits = np.split(class_indices, split_points)

        for i, split in enumerate(class_splits):
            partitions[i].extend(split.tolist())

    rng.shuffle(partitions[partition_id])
    return partitions[partition_id]

def load_data(partition_id: int, num_partitions: int, batch_size: int = 64,
              partition_mode: str = "iid", alpha: float = 0.5):
    """Carica FER-2013 con partizionamento IID o non-IID selezionabile."""
    transform = Compose([
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])

    trainset = ImageFolder(root="/home/picocluster/flower_node/data/train", transform=transform)
    testset = ImageFolder(root="/home/picocluster/flower_node/data/test", transform=transform)

    if partition_mode == "iid":
        train_indices = _iid_split(len(trainset), num_partitions, partition_id)
        test_indices = _iid_split(len(testset), num_partitions, partition_id)
    elif partition_mode == "non_iid":
        train_indices = _dirichlet_split(trainset.targets, num_partitions, partition_id, alpha=alpha)
        test_indices = _dirichlet_split(testset.targets, num_partitions, partition_id, alpha=alpha)
    else:
        raise ValueError(f"partition_mode sconosciuto: {partition_mode!r} (usa 'iid' o 'non_iid')")

    generator = torch.Generator()
    generator.manual_seed(SEED)

    trainloader = DataLoader(
        Subset(trainset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator
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