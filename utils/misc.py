import torch
def check_device():
    """
    Check the available device for running PyTorch operations.

    Returns:
        str: The name of the device ('mps', 'cuda', or 'cpu').
    """
    # Check if Multi-Process Service (MPS) is available
    if torch.backends.mps.is_available():
        device = 'mps'
    # Check if CUDA (GPU) is available
    elif torch.cuda.is_available():
        device = 'cuda'
    # If neither MPS nor CUDA is available, use the CPU
    else:
        device = 'cpu'

    # Print the selected device for information
    print('Working on Platform:', device)

    return device

def get_optimizer(optimizer, model, learning_rate):
    if optimizer =='Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer =='RMSProp':
        return torch.optim.RMSProp(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'The given optimizer is not set for this module.')


