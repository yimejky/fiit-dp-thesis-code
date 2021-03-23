import logging

import torch

from src.dataset.dataset_transforms import get_norm_transform
from src.dataset.transform_input import transform_input


def get_rescaled_pred(model, dataset, device, index, use_only_one_dimension=False,
                      transform_input_fn=transform_input):
    raw_data, raw_label = dataset.get_raw_item_with_label_filter(index)
    norm_data, norm_label = transform_input_fn(raw_data, raw_label, get_norm_transform())

    data_input = [norm_data[0]] if use_only_one_dimension else norm_data
    data_input = torch.tensor([data_input]).to(device).float()
    # data_input.shape => batch, channel, slices, x, y

    # removing batch dimension
    prediction = model(data_input)[0]
    del data_input

    prediction = prediction.cpu().detach().numpy()
    log_msg = f'get_rescaled_pred_0: {prediction.min()}, {prediction.max()}'
    logging.debug(log_msg)

    rescaled_pred = prediction - prediction.min()
    rescaled_pred = rescaled_pred / rescaled_pred.max()

    return prediction, rescaled_pred


def get_rescaled_preds(model, dataset, device,
                       transform_input_fn=transform_input):
    preds = []
    rescaled_preds = []
    for index in range(len(dataset)):
        prediction, rescaled_pred = get_rescaled_pred(model, dataset, device, index,
                                                      transform_input_fn=transform_input_fn)
        preds.append(prediction)
        rescaled_preds.append(rescaled_pred)

        del prediction
        del rescaled_pred
        torch.cuda.empty_cache()

    return preds, rescaled_preds


def get_raw_with_prediction(model, dataset, device, index, transform_input_fn=transform_input):
    raw_prediction, rescaled_pred = get_rescaled_pred(model, dataset, device, index,
                                                      transform_input_fn=transform_input_fn)
    raw_data, raw_label = dataset.get_raw_item_with_label_filter(index)
    return raw_data, raw_label, raw_prediction