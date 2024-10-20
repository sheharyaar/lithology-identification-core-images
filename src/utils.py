import torch

# Function to save the model checkpoint
def save_checkpoint(state, filename="best_model.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Function to load the model checkpoint
def load_checkpoint(filename="best_model.pth"):
    print("=> Loading checkpoint")
    return torch.load(filename)
