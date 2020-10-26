from operator import itemgetter

import torch

from src.model_and_training import ModelInfo
from src.model_and_training.getters.get_criterion import get_criterion
from src.model_and_training.getters.get_device import get_device
from src.model_and_training.getters.get_loaders import get_loaders
from src.model_and_training.getters.get_model import get_model
from src.model_and_training.getters.get_model_params import get_model_params
from src.model_and_training.getters.get_optimizer import get_optimizer
from src.model_and_training.getters.get_tensorboard_writer import get_tensorboard_writer


def load_checkpoint_model_info(model_name, epoch, train_dataset, valid_dataset, test_dataset):
    checkpoint_file_name = f'checkpoint_epoch_{epoch}.pkl'
    model_checkpoint_path = f'models/{model_name}/{checkpoint_file_name}'

    # Setting up model and stuff
    device = get_device()

    # loading checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    learning_rate, epochs, train_batch_size = itemgetter('learning_rate', 'epochs', 'train_batch_size')(checkpoint)
    model_params = checkpoint['model_params']

    # Architecture and optimizer with state
    model = get_model(device, model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    optimizer = get_optimizer(model=model, learning_rate=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = get_criterion()

    # Data loaders
    train_dataloader, valid_dataloader, test_dataloader = get_loaders(train_batch_size,
                                                                      train_dataset, valid_dataset, test_dataset)
    # Number of params
    model_total_params, model_total_trainable_params = get_model_params(model)
    # Tensorboard logs
    tensorboard_writer = get_tensorboard_writer(model_name)

    model_info: ModelInfo = {
        "model": model,
        "model_params": model_params,
        "model_name": model_name,
        "optimizer": optimizer,
        "criterion": criterion,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "device": device,
        "tensorboard_writer": tensorboard_writer,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader,
        "model_total_params": model_total_params,
        "model_total_trainable_params": model_total_trainable_params
    }

    return model_info
