import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import tqdm
from app import extract_attributes

# Set environment variable to optimize PyTorch memory usage
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Append necessary paths
sys.path.append('Models')
sys.path.append('utils')


def get_data(data):
    return data

# Custom imports
from unet_model import UNET
from evaluate import get_dataloader, val_loss_trn, train_batch, accuracy
from dice_loss import dice_loss
from misc import check_device, get_optimizer

def setup_model_optimizer(data_channels, optimizer_name, learning_rate, device):
    # Create the UNET model and move it to the specified device
    model = UNET(data_channels).to(device)
    # Configure optimizer and learning rate scheduler
    optimizer = get_optimizer(optimizer_name, model, learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5)

    return model, optimizer, lr_scheduler
data_channels = extract_attributes()[1]['n_channels']
optimizer_name = extract_attributes()[2]['optimizer']
learning_rate = extract_attributes()[2]['learning_rate']
device = check_device()
model, optimizer, lr_scheduler = setup_model_optimizer(data_channels, optimizer_name, learning_rate, device)


def train_fun(model, epochs, train_ldr, test_ldr):
    """
    Train the UNET model and evaluate on validation set.

    Args:
        model (torch.nn.Module): The UNET model to be trained.
        epochs (int): Number of training epochs.
        train_ldr (torch.utils.data.DataLoader): DataLoader for training data.
        test_ldr (torch.utils.data.DataLoader): DataLoader for validation data.

    Returns:
        tuple: Tuple containing lists of training loss, training accuracies,
               validation loss, and validation accuracies.
    """
    train_loss = []
    train_accuracies = []
    val_loss = []
    val_accuracies = []

    # Loop over epochs
    for i in tqdm.tqdm(range(epochs), total=epochs):
        print(f'Epoch: {i}')
        train_epoch_losses = []
        train_epoch_accuracies = []
        val_epoch_losses = []
        val_epoch_accuracies = []

        # Training loop
        for ix, batch in tqdm.tqdm(enumerate(iter(train_ldr)), total=len(train_ldr)):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            train_epoch_losses.append(train_batch(x, y, model, optimizer))
        train_epoch_loss = np.mean(train_epoch_losses)
        print(f'Epoch: {i} Training Loss: {train_epoch_loss}')
        # Update learning rate using the scheduler
        lr_scheduler.step()  # Update learning rate

        # Training accuracy
        for ix, batch in tqdm.tqdm(enumerate(iter(train_ldr)), total=len(train_ldr)):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            train_epoch_accuracies.append(accuracy(x, y, model))
        train_epoch_accuracy = np.mean(train_epoch_accuracies)
        print(f'Epoch: {i} Training Accuracy: {train_epoch_accuracy}')

        # Validation loss
        for ix, batch in tqdm.tqdm(enumerate(iter(test_ldr)), total=len(test_ldr)):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            val_epoch_losses.append(val_loss_trn(x, y, model))
        val_epoch_loss = np.mean(val_epoch_losses)
        print(f'Epoch: {i} Validation Loss: {val_epoch_loss}')

        # Validation accuracy
        for ix, batch in tqdm.tqdm(enumerate(iter(test_ldr)), total=len(test_ldr)):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            val_epoch_accuracies.append(accuracy(x, y, model))
        val_epoch_accuracy = np.mean(val_epoch_accuracies)
        print(f'Epoch: {i} Validation Accuracy: {val_epoch_accuracy}')

        # Store metrics for plotting
        train_loss.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

    # Save trained model
    torch.save(model.state_dict(), 'outputs/model/model.pth')

    loss_train = np.array(train_loss, dtype=np.float32).mean()
    acc_train = np.array(train_accuracies, dtype=np.float32).mean()
    loss_val = np.array(val_loss, dtype=np.float32).mean()
    acc_val = np.array(val_accuracies, dtype=np.float32).mean()

    data = {
    "train_loss": float(loss_train),
    "train_accuracies": float(acc_train),
    "val_loss": float(loss_val),
    "val_accuracies": float(acc_val)
    }
    output_file_path = 'outputs/report/output_values.json'
    json_data = json.dumps(data)
    with open(output_file_path, 'w') as json_file:
         json_file.write(json_data)
    return train_loss, train_accuracies, val_loss, val_accuracies

def plot_metrics(train_values, val_values, epochs, ylabel, filename_prefix):
    """
    Plot training and validation metrics (loss or accuracy).

    Args:
        train_values (list): List of training metric values for each epoch.
        val_values (list): List of validation metric values for each epoch.
        epochs (np.ndarray): Array of epoch numbers.
        ylabel (str): Label for the y-axis ('loss' or 'accuracy').
        filename_prefix (str): Prefix for the filename to save the plot.
    """
    plt.plot(epochs, train_values, label='Training')
    plt.plot(epochs, val_values, label='Validation')
    plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(1))
    plt.title(f'Training and Validation {ylabel}')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()

    if ylabel == 'accuracy':
        plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])
    elif ylabel == 'loss':
        plt.gca().set_yticklabels(['{:.0f}'.format(x * 1) for x in plt.gca().get_yticks()])

    plt.grid(False)
    plt.savefig(f'outputs/graphs/{filename_prefix}_{ylabel}.jpg')

def generate_graph(epochs, train_ldr, test_ldr):
    """
    Generate training and validation loss plots.

    Args:
        epochs (int): Number of training epochs.
    """
    train_loss, train_accuracies, val_loss, val_accuracies = train_fun(model, epochs, train_ldr, test_ldr)
    epochs = np.arange(epochs) + 1
    plot_metrics(train_loss, val_loss, epochs, 'loss', 'train_val')



