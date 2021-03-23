import torch
import torchio
import numpy as np
import logging


def get_torchio_subject(item_data, item_label):
    # converting to inputs to torchio img
    t1 = torchio.ScalarImage(tensor=item_data)
    label = torchio.LabelMap(tensor=item_label)
    subject = torchio.Subject(t1=t1, label=label)

    return subject


def transform_input(item_data, item_label, transform):
    subject = get_torchio_subject(item_data, item_label)

    trans_subject = transform(subject)

    # converting back
    item_data = trans_subject.t1.numpy()
    item_label = trans_subject.label.numpy().astype(np.int8)

    return item_data, item_label


def get_torchio_registration_subject(item_data, item_label):
    log_msg = f'get_torchio_registration_subject_0: {item_data.shape}, {item_label.shape}'
    # print(log_msg)
    logging.debug(log_msg)

    # converting to inputs to torchio img
    if not torch.is_tensor(item_data):
        item_data = torch.tensor(item_data)
        item_label = torch.tensor(item_label)

    t1_input = torch.unsqueeze(item_data[0], 0)
    t2_input = torch.unsqueeze(item_data[1], 0)
    # print(f"{t1_input.shape}", {t2_input.shape})

    t1 = torchio.ScalarImage(tensor=t1_input)
    t2 = torchio.LabelMap(tensor=t2_input)
    label = torchio.LabelMap(tensor=item_label)
    subject = torchio.Subject(t1=t1, t2=t2, label=label)

    return subject


def transform_input_with_registration(item_data, item_label, transform):
    # parsing to subject
    subject = get_torchio_registration_subject(item_data, item_label)

    # transform
    trans_subject = transform(subject)

    # converting back
    item_data_t1 = trans_subject.t1.numpy()[0]
    item_data_t2 = trans_subject.t2.numpy()[0]
    # print(f"item_data t1,t2: {item_data_t1.shape}", {item_data_t2.shape})
    item_data = np.array([item_data_t1, item_data_t2])
    # print(f"item_data: {item_data.shape}")

    item_label = trans_subject.label.numpy().astype(np.int8)

    return item_data, item_label
