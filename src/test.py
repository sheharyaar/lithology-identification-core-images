import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import LithologyCNN
from utils import load_checkpoint  # Import the load_checkpoint function
from tqdm import tqdm  # Import tqdm for progress bar
import json

# Function to save the predictions to a JSON file
def save_predictions(true_labels, predicted_labels, filename='predictions.json'):
    with open(filename, 'w') as f:
        json.dump({
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        }, f)
    print(f'Predictions saved to {filename}')

# Function to test the model on the test dataset
def test_model(model, test_loader, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    
    # Add tqdm progress bar for the test loop
    test_loader_with_progress = tqdm(test_loader, desc="Testing", leave=False)

    with torch.no_grad():  # No need to compute gradients during testing
        for inputs, labels in test_loader_with_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Get predictions
            _, predicted = outputs.max(1)  # Get the predicted class
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()  # Count the number of correct predictions

            # Collect the true and predicted labels
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Update tqdm progress bar description with accuracy
            test_loader_with_progress.set_postfix({
                "accuracy": 100. * correct / total
            })

    # Print test accuracy after completing all batches
    print(f'Test Accuracy: {100.*correct/total}')

    # Save predictions for the report generation
    save_predictions(true_labels, predicted_labels)

if __name__ == "__main__":
    # Define image transformations for the test dataset
    transform = transforms.Compose([transforms.Resize((224, 224)),  # Resize to 224x224
                                    transforms.ToTensor()])  # Convert to tensor
    
    # Load the test dataset
    test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=12)  # No need to shuffle test data
    
    # Initialize the model and move it to CPU (or CUDA)
    model = LithologyCNN(num_classes=4).to('cuda')

    # Load the saved model checkpoint
    load_checkpoint(model, filename='best_model.pth', device='cuda')
    
    # Test the model
    test_model(model, test_loader, device='cuda')
