from torch.utils.tensorboard import SummaryWriter


def get_tensorboard_writer(model_name):
    tensorboard_writer = SummaryWriter(log_dir=f'logs/{model_name}')

    return tensorboard_writer
