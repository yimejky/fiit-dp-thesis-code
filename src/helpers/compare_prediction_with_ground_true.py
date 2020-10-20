import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display
from ipywidgets import widgets


def compare_prediction_with_ground_true(dataset, prediction,
                                        dataset_index, pred_threshold=0.5,
                                        max_slices=None, default_slice=None):
    if max_slices is None:
        max_slices = len(prediction[dataset_index])
    if default_slice is None:
        default_slice = max_slices // 2

    tmp_pure_prediction = prediction[dataset_index]
    tmp_data, tmp_label = dataset[dataset_index]
    tmp_data = tmp_data[0]  # removing channel dimension
    tmp_pred = ((tmp_pure_prediction > pred_threshold) * 1).astype(np.int8)

    intersection = tmp_pred * tmp_label

    empty_compare_img = np.zeros((*tmp_data.shape, 3))
    empty_compare_img[:, :, :, 0] = tmp_label - intersection
    empty_compare_img[:, :, :, 1] = intersection
    empty_compare_img[:, :, :, 2] = tmp_pred - intersection

    data_compare_img = np.stack((tmp_data,) * 3, axis=-1)
    data_compare_img = data_compare_img - data_compare_img.min()
    data_compare_img = data_compare_img / data_compare_img.max()
    tmp_cond = empty_compare_img > 0
    data_compare_img[tmp_cond] = empty_compare_img[tmp_cond]

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
        plt.imshow(tmp_pred[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(2, 3, 6).set_title('prediction float mask')
        plt.imshow(tmp_pure_prediction[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.show()

    aSlider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=default_slice)
    ui = widgets.VBox([widgets.HBox([aSlider])])
    out = widgets.interactive_output(f, {'slice_index': aSlider})
    display(ui, out)


