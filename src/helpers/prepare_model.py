import sys
import numpy as np
import torch
import datetime

from datetime import timedelta
from time import time
from contextlib import nullcontext
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.helpers.helpers import calc_dsc
from src.dice_loss import DiceLoss
from src.unet_architecture import UNet
from src.unet_architecture_v2 import UNetV2

IN_COLAB = 'google.colab' in sys.modules



def prepare_model(epochs=30, # 50 x train_size = number of steps, 200 with lr=5e-3 is enough
                     learning_rate=5e-3, 
                     train_dataset=None, valid_dataset=None, test_dataset=None):
    ### Params
    log_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f'{log_date}_3d_unet'

    criterion = DiceLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()


    ### Setting up model and stuff
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device running "{device}"')
    torch.cuda.empty_cache()

    # Architecture and optimizer
    model = UNet().to(device)
    # model = UNetV2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dataloaders
    num_workers = os.cpu_count() if IN_COLAB else 0
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    ### Tensorboard logs
    def get_indices(dataset):
        return str(sorted([i+1 for i in dataset.indices]))

    tensorboard_writer = SummaryWriter(log_dir=f'logs/{model_name}')
    tensorboard_writer.add_text('data_indices_train', get_indices(train_dataset))
    tensorboard_writer.add_text('data_indices_valid', get_indices(valid_dataset))
    tensorboard_writer.add_text('data_indices_test', get_indices(test_dataset))
    tensorboard_writer.add_text('optimizer_learning_rate', str(learning_rate))
    tensorboard_writer.add_text('epochs', str(epochs))
    tensorboard_writer.add_text('loss_function', str(type(criterion).__name__))

    return {
        "model": model,
        "model_name": model_name,
        "optimizer": optimizer,
        "criterion": criterion,
        "epochs": epochs,
        "device": device,
        "tensorboard_writer": tensorboard_writer,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader
    }
