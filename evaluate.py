import sys
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, f1_score
import matplotlib.pyplot as plt

sys.path.append('utils')
from load_data import get_customDataset
from dice_loss import dice_loss


def get_dataloader(image_folder_path='', csv='', csv_path='', csv_train_path='', csv_test_path='', image_folder='', split='', Batch_Size='', Test_Batch_Size=''):
    """
    Get DataLoader instances for training and testing data.

    Args:
        image_folder_path (str): Path to image folder for dataset.
        csv (bool): Whether to use CSV dataset or not.
        csv_path (str): Path to CSV file.
        csv_train_path (str): Path to CSV file for training data.
        csv_test_path (str): Path to CSV file for testing data.
        image_folder (bool): Whether to use image folder for dataset or not.
        split (bool): Whether to split data into training and testing sets.

    Returns:
        DataLoader: DataLoader instance for training data.
        DataLoader: DataLoader instance for testing data.
    """
    if split and image_folder: 
        data_train, data_test = get_customDataset(image_folder_path=image_folder_path, csv=csv, csv_path=csv_path, image_folder=image_folder, split=split)
    elif split and csv:
        data_train, data_test = get_customDataset(image_folder_path='', csv=csv, csv_path=csv_path, image_folder=image_folder, split=split)
    else:
        data_train = get_customDataset(csv=True, csv_path=csv_train_path, split=False)
        data_test = get_customDataset(csv=True, csv_path=csv_test_path, split=False)

    train_ldr = DataLoader(data_train, batch_size=Batch_Size, drop_last=False, shuffle=True, num_workers=0)
    test_ldr = DataLoader(data_test, batch_size=Test_Batch_Size, drop_last=False, num_workers=0)

    return train_ldr, test_ldr

def train_batch(x, y, model, optimizer):
    """
    Train a batch using the given model and optimizer.

    Args:
        x (torch.Tensor): Input data.
        y (torch.Tensor): Ground truth data.
        model (nn.Module): The model to train.
        optimizer: The optimizer to use for training.

    Returns:
        float: Batch loss.
    """
    model.train()
    prediction = model(x)
    batch_loss = dice_loss(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

def calculate_iou(predicted_masks, ground_truth_masks):
    """
    Calculate the Intersection over Union (IoU) scores.

    Args:
        predicted_masks (torch.Tensor): Predicted masks.
        ground_truth_masks (torch.Tensor): Ground truth masks.

    Returns:
        float: Average IoU score.
    """
    iou_scores = []  
    for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
        intersection = torch.logical_and(pred_mask, gt_mask).sum()
        union = torch.logical_or(pred_mask, gt_mask).sum()    
        iou = intersection / union     
        iou_scores.append(iou)
    avg_iou =  sum(iou_scores) / len(iou_scores)
    return avg_iou

@torch.no_grad()
def accuracy(x, y, model):
    """
    Calculate accuracy (IoU) of the model's predictions.

    Args:
        x (torch.Tensor): Input data.
        y (torch.Tensor): Ground truth data.
        model (nn.Module): The model to evaluate.

    Returns:
        float: Accuracy (IoU) score.
    """
    model.eval()
    pred = model(x)
    accuracy = calculate_iou(pred, y)
    return accuracy.detach().cpu().numpy()

@torch.no_grad()
def val_loss_trn(x, y, model):
    """
    Calculate validation loss for the training data.

    Args:
        x (torch.Tensor): Input data.
        y (torch.Tensor): Ground truth data.
        model (nn.Module): The model to evaluate.

    Returns:
        float: Validation loss.
    """
    prediction = model(x)
    val_loss = dice_loss(prediction, y)
    return val_loss.item()


