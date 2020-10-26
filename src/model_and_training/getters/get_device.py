import torch


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device running "{device}"')
    torch.cuda.empty_cache()
    return device
