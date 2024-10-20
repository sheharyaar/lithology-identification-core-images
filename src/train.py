import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LithologyCNN
import torch.nn as nn

# Function to train the model
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cuda'):
    # Define loss function: Cross Entropy Loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer, which is commonly used for deep learning tasks
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loop through the number of epochs
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Loop through the batches of data from the training set
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass: Get predictions
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass: Compute gradients
            optimizer.step()  # Update the weights
            
            # Calculate running loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Get predicted class
            total += labels.size(0)  # Track total number of samples
            correct += predicted.eq(labels).sum().item()  # Track number of correct predictions
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {100.*correct/total}')
        
        # After each epoch, validate the model
        validate_model(model, val_loader, device)

# Function to validate the model on the validation set
def validate_model(model, val_loader, device='cuda'):
    model.eval()  # Set model to evaluation mode (no gradients will be computed)
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients in validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Get predictions
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Print validation accuracy
    print(f'Validation Accuracy: {100.*correct/total}')
    
if __name__ == "__main__":
    # Define image transformations for the training and validation datasets
    transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize images to 224x224
                                    transforms.ToTensor()])  # Convert images to PyTorch tensors
    
    # Load training and validation datasets from directories
    train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='dataset/val', transform=transform)
    
    # Define DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle data to improve training
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data
    
    # Initialize the model and move it to GPU (if available)
    model = LithologyCNN(num_classes=4).to('cuda')
    
    # Train the model
    train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cuda')
