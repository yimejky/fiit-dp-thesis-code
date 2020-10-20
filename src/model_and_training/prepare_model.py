import os
import torch
import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.consts import IN_COLAB
from src.losses.dice_loss import DiceLoss
from src.model_and_training.unet_architecture_v2 import UNetV2


def prepare_model(epochs=30,  # 30 x train_size = number of steps
                  learning_rate=5e-4,
                  in_channels=16,
                  batch_size=1,
                  train_dataset=None, valid_dataset=None, test_dataset=None):
    # Params
    log_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f'{log_date}_3d_unet'

    criterion = DiceLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()

    # Setting up model and stuff
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device running "{device}"')
    torch.cuda.empty_cache()

    # Architecture and optimizer
    # model = UNet(in_channels=in_channels).to(device)
    model = UNetV2(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Data loaders
    num_workers = os.cpu_count() if IN_COLAB else 0
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # Number of params
    model_total_params = sum(p.numel() for p in model.parameters())
    model_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Tensorboard logs
    def get_indices(dataset):
        return str(sorted([i + 1 for i in dataset.indices]))

    tensorboard_writer = SummaryWriter(log_dir=f'logs/{model_name}')
    tensorboard_writer.add_text('data_indices_train', get_indices(train_dataset))
    tensorboard_writer.add_text('data_indices_valid', get_indices(valid_dataset))
    tensorboard_writer.add_text('data_indices_test', get_indices(test_dataset))
    tensorboard_writer.add_text('optimizer_learning_rate', str(learning_rate))
    tensorboard_writer.add_text('epochs', str(epochs))
    tensorboard_writer.add_text('loss_function', str(type(criterion).__name__))
    tensorboard_writer.add_text('model_number_of_params', str(model_total_params))
    tensorboard_writer.add_text('model_number_of_trainable_params', str(model_total_trainable_params))

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
        "test_dataloader": test_dataloader,
        "model_total_params": model_total_params,
        "model_total_trainable_params": model_total_trainable_params
    }
