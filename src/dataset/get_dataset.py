import torch

from src.consts import IN_COLAB
from src.dataset.get_dataset_transform import get_dataset_transform
from src.dataset.han_oars_dataset import HaNOarsDataset


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

    # setting transform
    dataset.transform = get_dataset_transform()

    # processing
    if filter_labels is not None:
        dataset.filter_labels(filter_labels, unify_labels=unify_labels)

    return dataset
