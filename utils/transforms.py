import torchvision.transforms as T
import torch
import cv2
from app import extract_attributes


HEIGHT_WIDTH = extract_attributes()[1]['height']

def get_height_width():
    return HEIGHT_WIDTH


def transform_fun(image):
    """
    Apply a series of transformations to the input image.

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        torch.Tensor: The transformed image as a PyTorch tensor.
    """
    # Define a sequence of transformations using torchvision.transforms.Compose
    # This transforms the image into a PyTorch tensor, resizes it, and normalizes it.
    HEIGHT_WIDTH = get_height_width()
    transform = T.Compose([
        T.ToPILImage(),          # Convert the NumPy array to a PIL image
        T.ToTensor(),            # Convert the PIL image to a PyTorch tensor (values scaled between 0 and 1)
        T.Resize(HEIGHT_WIDTH)   # Resize the tensor to a height and width of 224 pixels
    ])

    # Apply the defined transformations to the input image
    transformed_image = transform(image)

    return transformed_image
