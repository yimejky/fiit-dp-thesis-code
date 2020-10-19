import torch
import numpy as np
import pandas as pd

from src.helpers.calc_dsc import calc_dsc


def get_rescaled_preds(model, dataset, device):
    preds = []
    rescaled_preds = []
    for i in range(len(dataset)):
        data, label = dataset[i]
        data_input = torch.from_numpy(np.array([data])).to(device).float()
        # data_input.shape => batch, channel, slices, x, y

        prediction = model(data_input)[0]
        prediction = prediction.cpu().detach().numpy()[0]
        rescaled_pred = prediction - prediction.min()
        rescaled_pred = rescaled_pred / rescaled_pred.max()

        preds.append(prediction)
        rescaled_preds.append(rescaled_pred)

    return preds, rescaled_preds,


def get_dataset_threshold_info(dataset, preds, rescaled_preds, index, info_list, is_train=False, is_valid=False,
                               is_test=False):
    data, label = dataset[index]
    prediction = preds[index]
    rescaled_pred = rescaled_preds[index]

    info = {}
    info['index'] = index
    info['dsc'] = calc_dsc(label, prediction)
    info['rescaled_dsc'] = calc_dsc(label, rescaled_pred)
    info['is_train'] = is_train
    info['is_valid'] = is_valid
    info['is_test'] = is_test

    step = 0.01
    for thresh in np.arange(0, 1 + step, step):
        tmp_text = "{:.2f}".format(thresh)
        info[f'thres_rescaled_dsc_{tmp_text}'] = calc_dsc(label, (rescaled_pred > thresh) * 1)

    info_list.append(info)
    return info_list


def get_threshold_info_df(model, dataset, device, train_indices, valid_indices, test_indices):
    preds, rescaled_preds = get_rescaled_preds(model, dataset, device)
    info_list = []

    # get table with dsc, rescaled dsc and treshold dsc with some steps, with subset info
    print('starting calc dsc per threshold')
    for index in list(sorted(train_indices)):
        info_list = get_dataset_threshold_info(dataset, preds, rescaled_preds, index, info_list, is_train=True)
    print('done train')
    for index in list(sorted(valid_indices)):
        info_list = get_dataset_threshold_info(dataset, preds, rescaled_preds, index, info_list, is_valid=True)
    print('done valid')
    for index in list(sorted(test_indices)):
        info_list = get_dataset_threshold_info(dataset, preds, rescaled_preds, index, info_list, is_test=True)
    print('done test')
    info_df = pd.DataFrame(info_list).set_index('index')

    return info_df, preds, rescaled_preds
