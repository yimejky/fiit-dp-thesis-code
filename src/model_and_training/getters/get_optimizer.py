import torch


def get_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

