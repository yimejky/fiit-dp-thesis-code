from operator import itemgetter

import torch


def split_dataset(dataset, train_size=40, valid_size=5, test_size=5):
    # splitting 40:5:5
    sizes_list = [train_size, valid_size, test_size]
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, sizes_list)

    return {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "train_size": train_size,
        "valid_size": valid_size,
        "test_size": test_size
    }


def copy_split_dataset(new_dataset, original_dataloaders_obj):
    train_low_res_dataset, valid_low_res_dataset, test_low_res_dataset = itemgetter(
        'train_dataset', 'valid_dataset', 'test_dataset')(original_dataloaders_obj)

    train_dataset = torch.utils.data.Subset(new_dataset, train_low_res_dataset.indices)
    valid_dataset = torch.utils.data.Subset(new_dataset, valid_low_res_dataset.indices)
    test_dataset = torch.utils.data.Subset(new_dataset, test_low_res_dataset.indices)

    return {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "train_size": len(train_dataset),
        "valid_size": len(valid_dataset),
        "test_size": len(test_dataset)
    }

