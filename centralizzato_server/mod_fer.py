import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Normalize, ToTensor

# 1. Definizione dell'Architettura CNN (Identica a task.py)
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

# 2. Selezione del Device (MPS per Apple Silicon, CUDA o CPU)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# 3. Funzioni di Train ed Evaluation
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(dataloader), correct / total

# 4. Pipeline Principale
def main():
    # Iperparametri
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.01
    DATA_DIR = "./data"  # Assicurati che contenga /train e /test
    
    device = get_device()
    print(f"Utilizzo del device: {device}")
    
    # Pre-processing immagini FER-2013
    transform = Compose([
        Grayscale(num_output_channels=1),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    
    # Caricamento dataset centralizzato
    trainset = ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
    testset = ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Inizializzazione Modello, Loss e Ottimizzatore
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    print("\n--- Inizio Addestramento Centralizzato ---")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        
        print(f"Epoca [{epoch:02d}/{EPOCHS:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%")
        
    total_time = time.time() - start_time
    print(f"\nAddestramento completato in {total_time:.2f} secondi.")
    
    # Salvataggio pesi finali
    torch.save(model.state_dict(), "centralized_fer2013_model.pt")
    print("Modello salvato come 'centralized_fer2013_model.pt'")

if __name__ == "__main__":
    main()