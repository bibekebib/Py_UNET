import torch
import cv2
import sys
import matplotlib.pyplot as pl

# Add paths to required modules and utilities
sys.path.append('Models')
from unet_model import UNET
sys.path.append('utils')
# from misc import check_device
from transforms import transform_fun
from app import extract_attributes


def load_segmentation_model(model_path, n_channels):
    """
    Load and return the pretrained UNET segmentation model.

    Args:
        model_path (str): Path to the pretrained model weights.

    Returns:
        nn.Module: The loaded UNET model.
    """
    model = UNET(n_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_binary_mask(model, input_image_path, threshold=0.75):
    """
    Generate and save the predicted binary mask from the input image using the UNET model.

    Args:
        model (nn.Module): The trained UNET model.
        input_image_path (str): Path to the input image.
        threshold (float): Threshold for generating the binary mask. Default is 0.5.
    """
    # Transform the input image for model prediction
    input_image = transform_fun(cv2.imread(input_image_path))

    # Generate predictions using the trained model
    with torch.no_grad():
        predicted_masks = model(input_image.unsqueeze(0))
        predicted_masks = predicted_masks.squeeze().cpu().numpy()
    # print(predicted_masks)
    # Apply a threshold to generate a binary mask
    binary_mask = np.where(predicted_masks >= threshold, 0, 255).astype('uint8')
    # print(binary_mask)

    return binary_mask

def save_and_display_mask(binary_mask, output_mask_path):
    """
    Save the binary mask image and display it using matplotlib.

    Args:
        binary_mask (numpy.ndarray): Predicted binary mask as a NumPy array.
        output_mask_path (str): Path to save the binary mask image.
    """
    cv2.imwrite(output_mask_path, binary_mask)
    plt.imshow(binary_mask, cmap='gray')  # Display the binary mask using matplotlib
    plt.show()

def main():
    model_path = 'outputs/model/model.pth'
    # The line `input_image_path = 'archive/Forest Segmented/Forest
    # Segmented/images/950926_sat_82.jpg'` is assigning the path of the input image to the variable
    # `input_image_path`. This path is used later in the code to load the input image for generating
    # the binary mask.
    input_image_path = 'archive/Forest Segmented/Forest Segmented/images/950926_sat_82.jpg'
    output_mask_path = 'outputs/result_image/testimage.jpg'
    n_channels =extract_attributes()[1]['n_channels']
    # Load the segmentation model
    model = load_segmentation_model(model_path, n_channels)

    # Generate and save the binary mask
    binary_mask = generate_binary_mask(model, input_image_path)
    save_and_display_mask(binary_mask, output_mask_path)


main()