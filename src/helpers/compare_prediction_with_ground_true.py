import matplotlib.pyplot as plt
import numpy as np
import torch

from IPython.display import display
from ipywidgets import widgets

from src.dataset.get_norm_transform import get_norm_transform
from src.dataset.transform_input import transform_input
from src.helpers.calc_dsc import calc_dsc


def compare_prediction_with_ground_true(dataset, prediction,
                                        dataset_index, pred_threshold=0.5,
                                        max_slices=None, default_slice=None):
    if max_slices is None:
        max_slices = len(prediction[dataset_index])
    if default_slice is None:
        default_slice = max_slices // 2

    tmp_raw_prediction = prediction[dataset_index]
    tmp_data, tmp_label = dataset.get_raw_item_with_label_filter(dataset_index)
    tmp_data, tmp_label = transform_input(tmp_data, tmp_label, get_norm_transform())

    tmp_data = tmp_data[0]  # removing channel dimension
    tmp_thresh_pred = ((tmp_raw_prediction > pred_threshold) * 1).astype(np.int8)

    intersection = tmp_thresh_pred * tmp_label

    # compare img without background
    empty_compare_img = np.zeros((*tmp_data.shape, 3))
    empty_compare_img[:, :, :, 0] = tmp_label - intersection
    empty_compare_img[:, :, :, 1] = intersection
    empty_compare_img[:, :, :, 2] = tmp_thresh_pred - intersection

    # compare img with background
    data_compare_img = np.stack((tmp_data,) * 3, axis=-1)
    data_compare_img = data_compare_img - data_compare_img.min()
    data_compare_img = data_compare_img / data_compare_img.max()
    tmp_cond = empty_compare_img > 0
    data_compare_img[tmp_cond] = empty_compare_img[tmp_cond]

    tensor_tmp_label = torch.tensor(tmp_label)
    raw_dsc = calc_dsc(tensor_tmp_label, torch.tensor(tmp_raw_prediction))
    threshold_dsc = calc_dsc(tensor_tmp_label, torch.tensor(tmp_thresh_pred))
    print(f'raw prediction: min {tmp_raw_prediction.min()}, max {tmp_raw_prediction.max()}, dsc {raw_dsc}')
    print(f'threshold prediction: min {tmp_thresh_pred.min()}, max {tmp_thresh_pred.max()}, dsc {threshold_dsc}')

    def f(slice_index):
        plt.figure(figsize=(30, 20))
        plt.subplot(2, 3, 1).set_title('comparison')
        plt.imshow(empty_compare_img[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(2, 3, 2).set_title('input data+comparison')
        plt.imshow(data_compare_img[slice_index], cmap="gray")

        plt.subplot(2, 3, 3).set_title('input data')
        plt.imshow(tmp_data[slice_index], cmap="gray")

        plt.subplot(2, 3, 4).set_title('ground true')
        plt.imshow(tmp_label[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(2, 3, 5).set_title('prediction bit mask')
        plt.imshow(tmp_thresh_pred[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(2, 3, 6).set_title('prediction float mask')
        plt.imshow(tmp_raw_prediction[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.show()

    aSlider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=default_slice)
    ui = widgets.VBox([widgets.HBox([aSlider])])
    out = widgets.interactive_output(f, {'slice_index': aSlider})
    display(ui, out)


