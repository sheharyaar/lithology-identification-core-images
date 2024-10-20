import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LithologyCNN

# Function to test the model on the test dataset
def test_model(model, test_loader, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Get predictions
            _, predicted = outputs.max(1)  # Get the predicted class
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()  # Count the number of correct predictions
    
    # Print test accuracy
    print(f'Test Accuracy: {100.*correct/total}')

if __name__ == "__main__":
    # Define image transformations for the test dataset
    transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize to 224x224
                                    transforms.ToTensor()])  # Convert to tensor
    
    # Load the test dataset
    test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle test data
    
    # Load the model
    model = LithologyCNN(num_classes=4).to('cuda')
    model.load_state_dict(torch.load('best_model.pth'))  # Load the trained model parameters
    
    # Test the model
    test_model(model, test_loader, device='cuda')
