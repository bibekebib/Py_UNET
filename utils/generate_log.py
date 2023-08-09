import torch  # Import the PyTorch library
from torchsummary import summary  # Import the summary function from torchsummary
import sys  # Import the sys module for system-related operations
import os  # Import the os module for operating system-related operations

# Custom Imports
sys.path.append('Models')  # Append the 'Models' folder path to the system path
from unet_model import UNET  # Import the UNET model from the 'unet_model' module
from misc import check_device  # Import the check_device function from the 'misc' module
from transforms import get_height_width  # Import the get_height_width function from the 'transforms' module

# Get the device (CPU or GPU or MPS) on which to run the model
device = check_device()

# Get the height and width for the input image from the 'transforms' module
height = get_height_width()
width = get_height_width()

def gen_summary(device, height, width, RGB=True):
    """
    Generate the model summary and save it to a file.

    Parameters:
        device (str): The name of the device ('mps', 'cuda', or 'cpu') to run the model.
        height (int): The height of the input image for the model.
        width (int): The width of the input image for the model.
        RGB (bool): A flag indicating whether the input images are RGB (True) or grayscale (False).

    Returns:
        None
    """
    if RGB:
        # Create a UNET model with 3 input channels (RGB) and move it to the specified device (CPU or GPU or MPS)
        model = UNET(3).to(device)
        # Generate the summary for the RGB model and convert it to a string
        out_summary = str(summary(model, input_size=(3, height, width), batch_size=-1, device=str(device)))
    else:
        # Create a UNET model with 1 input channel (grayscale) and move it to the specified device (CPU or GPU or MPS)
        model = UNET(1).to(device)
        # Generate the summary for the grayscale model and convert it to a string
        out_summary = str(summary(model, input_size=(1, height, width), batch_size=-1, device=str(device)))

    # Save the summary string to a file named 'summary_log.md' in the 'outputs' folder
    with open('outputs/summary_log.md', 'w+') as f:
        f.write(out_summary)


