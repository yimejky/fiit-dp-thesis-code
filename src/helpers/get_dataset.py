import torch

from operator import itemgetter
from torch.utils.data import Dataset

from src.consts import IN_COLAB
from src.han_oars_dataset import HaNOarsDataset
import src.helpers.oars_labels_consts as OARS_LABELS


def get_dataset(dataset_size=50,
                dataset_folder_name='HaN_OAR',
                filter_labels=None,
                shrink_factor=None,
                unify_labels=True):
    # if filter_labels is None:
    #     filter_labels = [OARS_LABELS.EYE_L, OARS_LABELS.EYE_R, OARS_LABELS.LENS_L, OARS_LABELS.LENS_R]

    if IN_COLAB:
        dataset_shrink = 4 if shrink_factor is None else shrink_factor
        print(f'COLAB using {dataset_shrink}x dataset')
        dataset = HaNOarsDataset(
            f'/content/drive/My Drive/data/{dataset_folder_name}_shrink{dataset_shrink}x_padded160', dataset_size)
    elif torch.cuda.is_available():
        dataset_shrink = 8 if shrink_factor is None else shrink_factor
        print(f'CUDA using {dataset_shrink}x dataset')
        dataset = HaNOarsDataset(
            f'./data/{dataset_folder_name}_shrink{dataset_shrink}x_padded160', dataset_size)
    else:
        dataset_shrink = 16 if shrink_factor is None else shrink_factor
        print(f'CPU using {dataset_shrink}x dataset')
        dataset = HaNOarsDataset(
            f'./data/{dataset_folder_name}_shrink{dataset_shrink}x_padded160', dataset_size)

    # processing
    dataset.data_normalize()
    if filter_labels is not None:
        dataset.filter_labels2(filter_labels, unify_labels=unify_labels)

    return dataset


def get_dataloaders(dataset, train_size=40, valid_size=5, test_size=5):
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


def get_copy_dataloaders(new_dataset, original_dataloaders_obj):
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
