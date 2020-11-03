import matplotlib.pyplot as plt
import numpy as np
import torch

from IPython.display import display
from ipywidgets import widgets

from src.dataset.get_norm_transform import get_norm_transform
from src.dataset.transform_input import transform_input
from src.helpers.calc_dsc import calc_dsc


def compare_one_prediction_with_ground_true(raw_data,
                                            raw_label,
                                            raw_prediction,
                                            pred_threshold=0.5,
                                            max_slices=None,
                                            default_slice=None):
    if max_slices is None:
        max_slices = len(raw_prediction)
    if default_slice is None:
        default_slice = max_slices // 2

    raw_data, raw_label = transform_input(raw_data, raw_label, get_norm_transform())
    raw_data = raw_data[0]  # removing channel dimension
    tmp_thresh_pred = ((raw_prediction > pred_threshold) * 1).astype(np.int8)

    intersection = tmp_thresh_pred * raw_label

    # compare img without background
    empty_compare_img = np.zeros((*raw_data.shape, 3))
    empty_compare_img[:, :, :, 0] = raw_label - intersection
    empty_compare_img[:, :, :, 1] = intersection
    empty_compare_img[:, :, :, 2] = tmp_thresh_pred - intersection

    # compare img with background
    data_compare_img = np.stack((raw_data,) * 3, axis=-1)
    data_compare_img = data_compare_img - data_compare_img.min()
    data_compare_img = data_compare_img / data_compare_img.max()
    tmp_cond = empty_compare_img > 0
    data_compare_img[tmp_cond] = empty_compare_img[tmp_cond]

    tensor_raw_label = torch.tensor(raw_label)
    raw_dsc = calc_dsc(tensor_raw_label, torch.tensor(raw_prediction))
    threshold_dsc = calc_dsc(tensor_raw_label, torch.tensor(tmp_thresh_pred))
    print(f'raw prediction: min {round(float(raw_prediction.min()), 4)}, max {round(float(raw_prediction.max()), 4)}, dsc {round(float(raw_dsc), 4)}')
    print(f'threshold prediction: min {round(float(tmp_thresh_pred.min()), 4)}, max {round(float(tmp_thresh_pred.max()), 4)}, dsc {round(float(threshold_dsc), 4)}')

    def f(slice_index):
        plt.figure(figsize=(30, 20))
        plt.subplot(2, 3, 1).set_title('comparison')
        plt.imshow(empty_compare_img[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(2, 3, 2).set_title('input data+comparison')
        plt.imshow(data_compare_img[slice_index], cmap="gray")

        plt.subplot(2, 3, 3).set_title('input data')
        plt.imshow(raw_data[slice_index], cmap="gray")

        plt.subplot(2, 3, 4).set_title('ground true')
        plt.imshow(raw_label[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(2, 3, 5).set_title('prediction bit mask')
        plt.imshow(tmp_thresh_pred[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(2, 3, 6).set_title('prediction float mask')
        plt.imshow(raw_prediction[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.show()

    aSlider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=default_slice)
    ui = widgets.VBox([widgets.HBox([aSlider])])
    out = widgets.interactive_output(f, {'slice_index': aSlider})
    display(ui, out)


def compare_prediction_with_ground_true(dataset, 
                                        prediction,
                                        dataset_index, 
                                        pred_threshold=0.5,
                                        max_slices=None, 
                                        default_slice=None):
    raw_prediction = prediction[dataset_index]
    raw_data, raw_label = dataset.get_raw_item_with_label_filter(dataset_index)

    compare_one_prediction_with_ground_true(raw_data,
                                            raw_label,
                                            raw_prediction,
                                            pred_threshold=pred_threshold,
                                            max_slices=max_slices,
                                            default_slice=default_slice)



