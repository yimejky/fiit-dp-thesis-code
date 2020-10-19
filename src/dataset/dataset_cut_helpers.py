import numpy as np
import matplotlib.pyplot as plt
import torch

from IPython.display import display
from ipywidgets import widgets

from src.consts import DESIRE_BOUNDING_BOX_SIZE
from src.helpers.get_bounding_box import get_bounding_box_3D, get_bounding_box_3D_size, get_final_bounding_box_slice


def expand_image(input_img, expand_factor=16):  # input numpy shape (1, 1, MAX_PADDING_SLICES, x, x)
    expanded_input_img = np.repeat(np.repeat(input_img, expand_factor, axis=3), expand_factor, axis=4)
    return expanded_input_img


def debug_preview_low_expand(model_output_img, exp_model_output_img):
    # preview of 32x32 segmentation and his expanded 512x512 version
    def f(slice_index):
        model_output_img_percents = model_output_img[0, 0, slice_index].sum() / model_output_img[0, 0, slice_index].size
        exp_model_output_img_percents = exp_model_output_img[0, 0, slice_index].sum() / exp_model_output_img[
            0, 0, slice_index].size
        print(model_output_img_percents, exp_model_output_img_percents)

        plt.figure(figsize=(12, 12))

        plt.subplot(1, 2, 1)
        plt.imshow(model_output_img[0, 0, slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(1, 2, 2)
        plt.imshow(exp_model_output_img[0, 0, slice_index], cmap="gray", vmin=0, vmax=1)

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

        plt.subplot(1, 3, 1)
        plt.imshow(tmp_cut[slice_index], cmap="gray", vmin=0, vmax=1)

        plt.subplot(1, 3, 2)
        plt.imshow(data_cut[0, slice_index], cmap="gray")

        plt.subplot(1, 3, 3)
        plt.imshow(label_cut[slice_index])

        plt.show()

    slices_count = label_cut.shape[0] - 1
    sliceSlider = widgets.IntSlider(min=0, max=slices_count, step=1, value=slices_count // 2)
    ui = widgets.VBox([widgets.HBox([sliceSlider])])
    out = widgets.interactive_output(f, {'slice_index': sliceSlider})
    # noinspection PyTypeChecker
    display(ui, out)


def get_full_res_cut(
        low_res_model,
        low_res_data_img,
        full_res_data_img,
        full_res_label_img,
        low_res_mask_threshold,
        desire_bounding_box_size,
        show_debug=False):
    # getting low res segmentation
    exp_low_res_data_img = np.expand_dims(low_res_data_img, axis=0)
    model_output_img = low_res_model(torch.from_numpy(exp_low_res_data_img).float())
    model_output_img = model_output_img.cpu().detach().numpy()

    # normalizing model output
    model_output_img = model_output_img - model_output_img.min()
    model_output_img = model_output_img / model_output_img.max()

    # parsing low res float to int mask
    model_output_img = (model_output_img > low_res_mask_threshold) * 1  # shape (1, 1, 160, 32, 32)

    # expanding low res int mask to high res
    exp_model_output_img = expand_image(model_output_img, expand_factor=16)  # shape (1, 1, 160, 512, 512)

    # getting bounding box
    bounding_box = get_bounding_box_3D(exp_model_output_img[0][0])
    # DEBUG, showing localizator bounding box
    # desire_bounding_box_size = get_bounding_box_3D_size(*bounding_box)
    new_bounding_box = get_final_bounding_box_slice(bounding_box, desire_bounding_box_size)

    # getting bounding box cut
    data_cut = full_res_data_img[0, new_bounding_box[0]:new_bounding_box[1] + 1,
               new_bounding_box[2]:new_bounding_box[3] + 1, new_bounding_box[4]:new_bounding_box[5] + 1]
    label_cut = full_res_label_img[new_bounding_box[0]:new_bounding_box[1] + 1,
                new_bounding_box[2]:new_bounding_box[3] + 1, new_bounding_box[4]:new_bounding_box[5] + 1]

    # data must have channel shape, (1, slices, x, y)
    data_cut = np.expand_dims(data_cut, axis=0)

    cut_sum = label_cut.sum()
    full_res_sum = full_res_label_img.sum()
    print('debug, does cut and original label contain the same amount of pixels?', cut_sum == full_res_sum, cut_sum,
          full_res_sum)
    # assert cut_sum == full_res_sum

    # debug
    if show_debug:
        print('debug bounding box sizes',
              get_bounding_box_3D_size(*bounding_box),
              get_bounding_box_3D_size(*new_bounding_box))
        print('debug bounding boxes', bounding_box, new_bounding_box)
        debug_preview_low_expand(model_output_img, exp_model_output_img)
        debug_preview_cuts(exp_model_output_img, new_bounding_box, data_cut, label_cut)

    return data_cut, label_cut, new_bounding_box


def get_cut_lists(low_res_model, low_res_dataset, full_res_dataset, cut_full_res_dataset, low_res_mask_threshold=0.5):
    for i in range(len(full_res_dataset)):
        print(f'getting cut index {i}')
        low_res_data_img = low_res_dataset.data_list[i]
        full_res_data_img = full_res_dataset.data_list[i]
        full_res_label_img = full_res_dataset.label_list[i]

        data_cut, label_cut, new_bounding_box = get_full_res_cut(low_res_model, low_res_data_img,
                                                                 full_res_data_img, full_res_label_img,
                                                                 low_res_mask_threshold,
                                                                 DESIRE_BOUNDING_BOX_SIZE,
                                                                 show_debug=False)
        cut_full_res_dataset.data_list[i] = data_cut
        cut_full_res_dataset.label_list[i] = label_cut

    return cut_full_res_dataset
