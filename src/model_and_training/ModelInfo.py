
from typing import TypedDict, Dict

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


class ModelInfo(TypedDict):
    model: nn.Module
    model_params: Dict
    model_name: str
    optimizer: nn.Module
    criterion: nn.Module
    epochs: int
    learning_rate: float
    train_batch_size: int
    device: torch.device
    tensorboard_writer: SummaryWriter
    train_dataloader: DataLoader
    valid_dataloader: DataLoader
    test_dataloader: DataLoader
    model_total_params: int
    model_total_trainable_params: int
