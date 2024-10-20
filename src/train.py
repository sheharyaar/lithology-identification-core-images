import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LithologyCNN
import torch.nn as nn
from tqdm import tqdm
from utils import save_checkpoint
import json

# Function to train the model
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_accuracy = 0
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loader_with_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for inputs, labels in train_loader_with_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_loader_with_progress.set_postfix({
                "loss": running_loss / len(train_loader),
                "accuracy": 100. * correct / total
            })

        train_acc.append(100. * correct / total)
        train_loss.append(running_loss / len(train_loader))

        val_accuracy, val_loss_value = validate_model(model, val_loader, criterion, device)
        val_acc.append(val_accuracy)
        val_loss.append(val_loss_value)

        print(f'Epoch {epoch+1}/{epochs}, Train Acc: {train_acc[-1]:.2f}, Val Acc: {val_accuracy:.2f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model.state_dict(), "best_model.pth")

    # Save the metrics
    save_metrics(train_acc, val_acc, train_loss, val_loss)

def validate_model(model, val_loader, criterion, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100. * correct / total
    return val_accuracy, running_val_loss / len(val_loader)

# Function to save metrics to a file
def save_metrics(train_acc, val_acc, train_loss, val_loss, filename='metrics.json'):
    metrics = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    with open(filename, 'w') as f:
        json.dump(metrics, f)
    print(f'Metrics saved to {filename}')

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=12)

    model = LithologyCNN(num_classes=4).to('cuda')
    
    train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cuda')
