import torch

# Function to save the model checkpoint
def save_checkpoint(state, filename="best_model.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Function to load the model checkpoint
def load_checkpoint(model, filename="best_model.pth", device='cuda'):
    print("=> Loading checkpoint from", filename)
    model.load_state_dict(torch.load(filename, map_location=device))  # Use map_location for device-specific loading
    return model