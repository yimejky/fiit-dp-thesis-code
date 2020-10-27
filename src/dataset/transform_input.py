import torchio
import numpy as np


def get_torchio_subject(item_data, item_label):
    tmp_label = np.expand_dims(item_label, axis=0)

    # converting to inputs to torchio img
    t1 = torchio.ScalarImage(tensor=item_data)
    label = torchio.LabelMap(tensor=tmp_label)
    subject = torchio.Subject(t1=t1, label=label)

    return subject


def transform_subject_input(subject, transform):
    trans_subject = transform(subject)

    # converting back
    item_data = trans_subject.t1.numpy()
    item_label = trans_subject.label.numpy()[0].astype(np.int8)

    return item_data, item_label


def transform_input(item_data, item_label, transform):
    subject = get_torchio_subject(item_data, item_label)

    return transform_subject_input(subject, transform)
