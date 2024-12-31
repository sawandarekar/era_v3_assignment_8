import sys
import os
import torch
import pickle  # Import pickle for saving the model

def save_checkpoint(state, filename='checkpoint.pth'):
    """Save the model checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    """Load the model checkpoint."""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Check if the scheduler state dict is present
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state
        else:
            print("Warning: 'scheduler_state_dict' not found in checkpoint. Skipping scheduler loading.")
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf')  # Return epoch 0 and infinite loss if no checkpoint found

def save_model(model, filename='model.pt'):
    """Save the model in .pt format."""
    torch.save(model.state_dict(), filename)
    print(f"Model saved in .pt format to {filename}")

def save_model_pickle(model, filename='model.pkl'):
    """Save the model in pickle format."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved in pickle format to {filename}")