import torch
import torch.nn.functional as F

def dice_loss(predicted, target, epsilon=1e-7):
    """
    Calculate the Dice Loss for semantic segmentation.

    Parameters:
        predicted (torch.Tensor): Predicted segmentation mask as a torch.Tensor.
        target (torch.Tensor): Ground truth segmentation mask as a torch.Tensor.
        epsilon (float): A small value added to the denominator to avoid division by zero.

    Returns:
        torch.Tensor: The Dice Loss as a scalar torch.Tensor.
    """
    target = target.float() / target.max()  # Convert target to probabilities

    # Calculate the intersection of predicted and target masks using probabilities
    intersection = (predicted * target).sum()

    # Calculate the union of predicted and target masks using probabilities
    union = predicted.sum() + target.sum()

    # Calculate the Dice coefficient
    dice_coeff = (2.0 * intersection + epsilon) / (union + epsilon)

    # Calculate the Dice Loss as 1 - Dice coefficient
    dice_loss = 1.0 - dice_coeff

    return dice_loss






