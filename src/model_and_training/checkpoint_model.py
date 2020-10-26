from operator import itemgetter

import torch

from pathlib import Path


def checkpoint_model(epoch, model_info):
    model, model_name, optimizer, criterion = itemgetter('model', 'model_name', 'optimizer', 'criterion')(model_info)
    learning_rate, epochs, train_batch_size = itemgetter('learning_rate', 'epochs', 'train_batch_size')(model_info)
    model_params = model_info["model_params"]

    folder_path = f"models/{model_name}"
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    state = {
        "model_params": model_params,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'epochs': epochs,
        "learning_rate": learning_rate,
        'criterion': criterion,
        'train_batch_size': train_batch_size
    }

    checkpoint_file_name = f'checkpoint_epoch_{epoch}.pkl'
    torch.save(state, f'{folder_path}/{checkpoint_file_name}')
