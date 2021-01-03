import torch
import numpy as np

from src.dataset.get_norm_transform import get_norm_transform
from src.dataset.transform_input import transform_input


def get_rescaled_pred(model, dataset, device, index, use_only_one_dimension=False):
    raw_data, raw_label = dataset.get_raw_item_with_label_filter(index)
    norm_data, norm_label = transform_input(raw_data, raw_label, get_norm_transform())

    data_input = norm_data
    if use_only_one_dimension:
        data_input = [data_input[0]]  # removing all dimensions
    data_input = np.array([data_input])
    data_input = torch.from_numpy(data_input).to(device).float()
    # data_input.shape => batch, channel, slices, x, y

    # removing batch dimension
    prediction = model(data_input)[0]
    del data_input

    prediction = prediction.cpu().detach().numpy()
    rescaled_pred = prediction - prediction.min()
    rescaled_pred = rescaled_pred / rescaled_pred.max()

    return prediction, rescaled_pred


def get_rescaled_preds(model, dataset, device):
    preds = []
    rescaled_preds = []
    for index in range(len(dataset)):
        prediction, rescaled_pred = get_rescaled_pred(model, dataset, device, index)
        preds.append(prediction)
        rescaled_preds.append(rescaled_pred)

        del prediction
        del rescaled_pred
        torch.cuda.empty_cache()

    return preds, rescaled_preds
