from src.consts import CPU_COUNT
from torch.utils.data import DataLoader


def get_loaders(train_batch_size, train_dataset, valid_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=CPU_COUNT)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=CPU_COUNT)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=CPU_COUNT)

    return train_dataloader, valid_dataloader, test_dataloader
