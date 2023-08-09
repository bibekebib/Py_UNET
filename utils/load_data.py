# Library Imports
import torch
from tqdm import tqdm
from PIL import Image
import os
import glob
import pandas as pd
import numpy as np
import re
import cv2
from torch.utils.data import Dataset
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Custom Imports
from misc import check_device
from transforms import transform_fun

# Check device
device = check_device()


def load_data(csv_file):
    '''
    Inputs:
        csv_file: A normal Csv file with two columns, first one Image and second one mask.
        For eg:
            image,mask
            10452_sat_08.jpg,10452_mask_08.jpg
            10452_sat_18.jpg,10452_mask_18.jpg
    Outputs:
        df: A dataframe
    '''
    df = pd.read_csv(csv_file)
    return df
        


def find_matching_folders(folder_list, image=True):
    """
    Find the first matching folder in the given list based on the specified image flag.

    Parameters:
        folder_list (list): A list of folder names to search through.
        image (bool): If True, search for image-related folders; otherwise, search for mask-related folders.

    Returns:
        str: The first matching folder name, or None if no match is found.
    """
    if image:
        # If 'image' flag is True, look for folders with names containing 'image', 'images', or 'img'
        pattern = r'image|images|img'
    else:
        # If 'image' flag is False, look for folders with names containing 'msk', 'mask', or 'masks'
        pattern = r'msk|mask|masks'

    # Find the matching folders based on the specified pattern (case-insensitive search)
    matching_folders = [folder for folder in folder_list if re.search(pattern, folder, re.IGNORECASE)]

    if matching_folders:
        # Return the first matching folder found
        return matching_folders[0]
    else:
        # If no matching folders are found, return None
        return None



def make_csv(image_folder, output=True, split=True):
    """
    Create a DataFrame from the image and mask folders within the specified image_folder.

    Parameters:
        image_folder (str): The path to the main folder containing image and mask subfolders.
        output (bool): If True, save the resulting DataFrame to a CSV file named 'df_folder.csv' in the 'outputs' folder.

    Returns:
        pandas.DataFrame: The DataFrame containing image and mask file paths, or None if the image_folder does not exist.
    """
    try:
        if os.path.exists(image_folder):
            # Get a list of subfolders within the image_folder
            path = image_folder
            folders = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]

            # Find the image and mask subfolders within the main folder
            img_folder = find_matching_folders(folders)
            mask_folder = find_matching_folders(folders, image=False)

            # Get a list of image and mask file paths within their respective subfolders
            images = glob.glob(image_folder+"/"+img_folder+'/*')
            masks = glob.glob(image_folder+"/"+mask_folder+'/*')

            # Create a DataFrame using the image and mask file paths
            df = pd.DataFrame(zip(images, masks), columns=['image', 'mask'])

            if output and not split:
                # If output is True, save the DataFrame to a CSV file named 'df_folder.csv' in the 'outputs' folder
                if os.path.exists('outputs'):
                    df.to_csv('outputs/df_folder.csv')
                else:
                    os.mkdir('outputs')
                    df.to_csv('outputs/df_folder.csv')
            if output and split:
                # If output is True, save the DataFrame to a CSV file named 'df_folder.csv' in the 'outputs' folder
                if os.path.exists('outputs'):
                    df_train, df_test = train_test_split(df, random_state=42)
                    df_train.to_csv('outputs/df_train_folder.csv')
                    df_test.to_csv('outputs/df_test_folder.csv')
                else:
                    os.mkdir('outputs')
                    df_train, df_test = train_test_split(df, random_state=42)
                    df_train.to_csv('outputs/df_train_folder.csv')
                    df_test.to_csv('outputs/df_test_folder.csv')      
                return df
            else:
                return df
        else:
            # If the specified image_folder does not exist, raise an exception
            raise FileNotFoundError("Folder Not Found: {}".format(image_folder))
    except FileNotFoundError as e:
        # Handle the exception and print the error message
        print(e)
        return None


def load_image(filename):
    """
    Load an image from the specified file.

    Parameters:
        filename (str): Path to the image file.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array (BGR format).
    """
    # Read the image using OpenCV's imread function (BGR format)
    image = cv2.imread(filename)

    # Note: If you need the image in RGB format, you can convert it here using cv2.cvtColor function:
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image






class CustomDataset(Dataset):
    def __init__(self, csv=True, csv_path='', image_folder=False, image_folder_path='', split=False, transform=True):
        """
        Custom Dataset class to load image and mask data.

        Parameters:
            csv (bool): If True, load image and mask file paths from a CSV file.
            csv_path (str): Path to the CSV file containing 'image' and 'mask' columns.
            image_folder (bool): If True, create a CSV file from image and mask subfolders within the specified folder.
            image_folder_path (str): Path to the main folder containing image and mask subfolders.
            transform (bool): If True, apply data transformations to images and masks.

        Attributes:
            image (list): List of image file paths.
            mask (list): List of mask file paths.
            transform (bool): Data transformation flag.
        """
        if csv:
            df = load_data(csv_path)
            self.image = df['image']
            self.mask = df['mask']
        elif image_folder:
            df = make_csv(image_folder_path, split=split)
            self.image = df['image']
            self.mask = df['mask']
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.image)

    def __getitem__(self, ix):
        """
        Retrieve a single sample from the dataset.

        Parameters:
            ix (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and corresponding mask as torch Tensors.
        """
        f = self.image[ix]
        m = self.mask[ix]
        image = load_image(f)
        mask = load_image(m)
        if self.transform:
            image = transform_fun(image)
            mask = transform_fun(image)  # Note: It should be transform_fun(mask) here, assuming the same transformation for both
        return torch.tensor(image).to(device), torch.tensor(mask).to(device)


def get_customDataset(image_folder_path='', csv='', csv_path='', image_folder='',  split=''):
    if not split:
        data = CustomDataset(csv=csv, csv_path=csv_path, image_folder=image_folder, image_folder_path=image_folder_path, split=split)
        return data
    else:
        data = CustomDataset(csv=csv, csv_path=csv_path, image_folder=image_folder, image_folder_path=image_folder_path, split=split)
        data_train, data_test = train_test_split(data, random_state=42)
    return data_train, data_test

