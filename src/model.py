import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNeXt50_32X4D_Weights

# Define the CNN model based on ResNeXt-50 architecture
class LithologyCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(LithologyCNN, self).__init__()
        # Load the pre-trained ResNeXt-50 model using the new 'weights' argument
        self.model = models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        # Replace the final fully connected layer to match our number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    # Define the forward pass
    def forward(self, x):
        return self.model(x)
