import torch
from torchvision import transforms
from PIL import Image
from model import LithologyCNN
from utils import load_checkpoint  # Import the load_checkpoint function
import os

# Define the preprocessing function for a single image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])
    image = Image.open(image_path)  # Open the image
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

# Function to make predictions on a single image
def predict_image(model, image_tensor, device='cuda'):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)  # Move image to device
        output = model(image_tensor)  # Forward pass
        _, predicted = output.max(1)  # Get the predicted class index
    return predicted.item()  # Return the predicted class index

# Function to map predicted class index to the actual class name
def get_class_name(class_idx, class_names):
    return class_names[class_idx]

if __name__ == "__main__":
    # Define class names based on your training classes
    class_names = ['garbage', 'limestone', 'sandstone', 'shale']

    # Path to your drill core images folder
    image_folder = './predict'  # Replace with the actual path to your images

    # Initialize the model and move it to GPU (or CPU if you don't have GPU)
    model = LithologyCNN(num_classes=4).to('cuda')

    # Load the saved model checkpoint
    load_checkpoint(model, filename='best_model.pth', device='cuda')

    # Loop through the images in the folder and make predictions
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('jpg', 'jpeg', 'png')):  # Process only image files
            image_path = os.path.join(image_folder, image_file)
            print(f"Processing {image_file}...")

            # Preprocess the image
            image_tensor = preprocess_image(image_path)

            # Make prediction
            predicted_class_idx = predict_image(model, image_tensor, device='cuda')
            predicted_class = get_class_name(predicted_class_idx, class_names)

            # Output the prediction
            print(f'Prediction for {image_file}: {predicted_class}')
