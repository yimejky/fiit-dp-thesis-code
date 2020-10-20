import torch

from pathlib import Path
from torchsummary import summary

from src.consts import IN_COLAB, MAX_PADDING_SLICES


def checkpoint_model(model_name, epoch, model, optimizer):
    folder_path = f"models/{model_name}"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, f'{folder_path}/checkpoint_{model_name}_epoch_{epoch}.pkl')


def show_model_info(model_info):
    if IN_COLAB:
        summary(model_info["model"], (1, MAX_PADDING_SLICES, 128, 128), batch_size=1)

    print(f'Model number of params: {model_info["model_total_params"]}, trainable {model_info["model_total_trainable_params"]}')
