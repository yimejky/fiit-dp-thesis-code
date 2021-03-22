import matplotlib.pyplot as plt

from IPython.display import display
from ipywidgets import widgets


def debug_preview_model_output(low_res_data_img, low_res_label_img, model_output_img):
    # preview of 32x32 segmentation and his expanded 512x512 version
    def f(slice_index):
        plt.figure(figsize=(18, 12))

        plt.subplot(1, 3, 1).set_title('low res data')
        plt.imshow(low_res_data_img[0, slice_index], cmap="gray")

        plt.subplot(1, 3, 2).set_title('low res label')
        plt.imshow(low_res_label_img[slice_index], cmap="gray")

        plt.subplot(1, 3, 3).set_title('low res model output')
        plt.imshow(model_output_img[0, 0, slice_index], cmap="gray", vmin=0, vmax=1)

        plt.show()

    slices_count = model_output_img[0, 0].shape[0] - 1
    sliceSlider = widgets.IntSlider(min=0, max=slices_count, step=1, value=slices_count // 2)
    ui = widgets.VBox([widgets.HBox([sliceSlider])])
    out = widgets.interactive_output(f, {'slice_index': sliceSlider})
    # noinspection PyTypeChecker
    display(ui, out)


def debug_preview_cuts(exp_model_output_img, new_bounding_box, data_cut, label_cut):
    def f(slice_index):
        tmp_cut = exp_model_output_img[0, 0, new_bounding_box[0]:new_bounding_box[1] + 1,
                  new_bounding_box[2]:new_bounding_box[3] + 1, new_bounding_box[4]:new_bounding_box[5] + 1]

        plt.figure(figsize=(18, 12))

        plt.subplot(1, 3, 1).set_title('low res model output')
        plt.imshow(tmp_cut[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(1, 3, 2).set_title('data cut')
        plt.imshow(data_cut[0, slice_index], cmap="gray")

        plt.subplot(1, 3, 3).set_title('label cut')
        plt.imshow(label_cut[slice_index])

        plt.show()

    slices_count = label_cut.shape[0] - 1
    sliceSlider = widgets.IntSlider(min=0, max=slices_count, step=1, value=slices_count // 2)
    ui = widgets.VBox([widgets.HBox([sliceSlider])])
    out = widgets.interactive_output(f, {'slice_index': sliceSlider})
    # noinspection PyTypeChecker
    display(ui, out)
