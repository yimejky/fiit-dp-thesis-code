import logging

import numpy as np
import pandas as pd
import torch

from src.dataset.get_norm_transform import get_norm_transform
from src.dataset.transform_input import transform_input
from src.helpers.calc_dsc import calc_dsc
from src.helpers.get_rescaled_pred import get_rescaled_preds


def get_dataset_threshold_info(dataset,
                               preds,
                               rescaled_preds,
                               index,
                               info_list,
                               is_train=False,
                               is_valid=False,
                               is_test=False,
                               step=0.01):
    raw_data, raw_label = dataset.get_raw_item_with_label_filter(index)
    norm_data, norm_label = transform_input(raw_data, raw_label, get_norm_transform())
    prediction = preds[index]
    rescaled_pred = rescaled_preds[index]

    tensor_norm_label = torch.tensor(norm_label)
    torch_prediction = torch.tensor(prediction)
    torch_rescaled_pred = torch.tensor(rescaled_pred)

    info = {'index': index,
            'dsc': calc_dsc(tensor_norm_label, torch_prediction).item(),
            'rescaled_dsc': calc_dsc(tensor_norm_label, torch_rescaled_pred).item(),
            'is_train': is_train,
            'is_valid': is_valid,
            'is_test': is_test}

    for thresh in np.arange(0, 1 + step, step):
        tmp_text = "{:.2f}".format(thresh)
        tensor_tresholded_rescaled = torch.tensor((rescaled_pred > thresh) * 1)
        info[f'thres_rescaled_dsc_{tmp_text}'] = calc_dsc(tensor_norm_label, tensor_tresholded_rescaled).item()

    info_list.append(info)
    return info_list


def get_threshold_info_df(model,
                          dataset,
                          device,
                          train_indices,
                          valid_indices,
                          test_indices,
                          step=0.01):
    logging.debug('get_threshold_info_df0 calculating all predictions')
    preds, rescaled_preds = get_rescaled_preds(model, dataset, device)
    info_list = []

    # get table with dsc, rescaled dsc and treshold dsc with some steps, with subset info
    logging.debug('get_threshold_info_df1 starting calc dsc per threshold')
    for index in list(sorted(train_indices)):
        info_list = get_dataset_threshold_info(dataset, preds, rescaled_preds, index, info_list,
                                               is_train=True,
                                               step=step)
    logging.debug('get_threshold_info_df2 done train')
    for index in list(sorted(valid_indices)):
        info_list = get_dataset_threshold_info(dataset, preds, rescaled_preds, index, info_list,
                                               is_valid=True,
                                               step=step)
    logging.debug('get_threshold_info_df3 done valid')
    for index in list(sorted(test_indices)):
        info_list = get_dataset_threshold_info(dataset, preds, rescaled_preds, index, info_list,
                                               is_test=True,
                                               step=step)
    logging.debug('get_threshold_info_df4 done test')
    info_df = pd.DataFrame(info_list).set_index('index')

    return info_df, preds, rescaled_preds
