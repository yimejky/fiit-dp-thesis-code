import os
import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import torch

from operator import itemgetter
from time import time
from pathlib import Path
from IPython.display import display, Markdown
from ipywidgets import interact, IntSlider
from ipywidgets import widgets
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.han_oars_dataset import HaNOarsDataset

import sys
IN_COLAB = 'google.colab' in sys.modules


def get_datasets(dataset_size = 50, dataset_folder_name = 'HaN_OAR_eyes+lens'):
    if (IN_COLAB):
        print('COLAB using 4x dataset')
        dataset = HaNOarsDataset(f'/content/drive/My Drive/data/{dataset_folder_name}_shrink4x_padded160', dataset_size)
    else:
        if torch.cuda.is_available():
            dataset_shrink = 16
            print(f'CUDA using {dataset_shrink}x dataset')
            dataset = HaNOarsDataset(f'./data/{dataset_folder_name}_shrink{dataset_shrink}x_padded160', dataset_size)
        else:
            print('CPU using 16x dataset')
            dataset = HaNOarsDataset(f'./data/{dataset_folder_name}_shrink16x_padded160', dataset_size)

    # 40:5:5
    train_size = 40
    valid_size = 5
    test_size = 5
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    
    return {
        "dataset": dataset,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size
    }
