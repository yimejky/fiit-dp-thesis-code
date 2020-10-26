import numpy as np
import torch

from src.dataset.debug_cut_helpers import debug_preview_cuts, debug_preview_model_output
from src.dataset.get_norm_transform import get_norm_transform
from src.dataset.transform_input import transform_input
from src.helpers.get_bounding_box import get_bounding_box_3D_size, get_final_bounding_box_slice, get_bounding_box_3D
from src.helpers.get_img_outliers_pixels import get_img_outliers_pixels


def expand_image(input_img, expand_factor=16):  # input numpy shape (1, 1, MAX_PADDING_SLICES, x, x)
    expanded_input_img = np.repeat(np.repeat(input_img, expand_factor, axis=3), expand_factor, axis=4)
    return expanded_input_img


def insert_to_model(low_res_model, low_res_data_img, low_res_mask_threshold, show_debug=False):
    # getting low res segmentation
    exp_low_res_data_img = np.expand_dims(low_res_data_img, axis=0)
    model_output_img = low_res_model(torch.from_numpy(exp_low_res_data_img).float())
    model_output_img = model_output_img.cpu().detach().numpy()

    # normalizing model output
    model_output_img = model_output_img - model_output_img.min()
    model_output_img = model_output_img / model_output_img.max()

    # parsing low res float to int mask
    model_output_img = (model_output_img > low_res_mask_threshold) * 1  # shape (1, 1, 160, 32, 32)
    model_output_img = model_output_img.astype(np.int8)

    return model_output_img


def get_full_res_cut(
        low_res_model,
        raw_low_res_data_img,
        raw_low_res_label_img,
        raw_full_res_data_img,
        raw_full_res_label_img,
        low_res_mask_threshold,
        desire_bounding_box_size,
        show_debug=False):
    # normalizing pure low res for model
    low_res_data_img, low_res_label_img = transform_input(raw_low_res_data_img,
                                                          raw_low_res_label_img,
                                                          get_norm_transform())

    # getting low res segmented image
    model_output_img = insert_to_model(low_res_model, low_res_data_img, low_res_mask_threshold, show_debug=show_debug)
    positive_pixels_count = np.sum(model_output_img > 0)

    remove_pixel_idx = get_img_outliers_pixels(model_output_img[0, 0])
    for x in remove_pixel_idx:
        model_output_img[tuple([0, 0, *x])] = 0
    print('debug removing', len(remove_pixel_idx), 'outlier pixels from', positive_pixels_count)

    # expanding low res int mask to high res
    exp_model_output_img = expand_image(model_output_img, expand_factor=16)  # shape (1, 1, 160, 512, 512)

    # getting bounding box
    bounding_box = get_bounding_box_3D(exp_model_output_img[0][0])
    # DEBUG, showing localizator bounding box
    # desire_bounding_box_size = get_bounding_box_3D_size(*bounding_box)
    new_bounding_box = get_final_bounding_box_slice(bounding_box, desire_bounding_box_size)

    # getting bounding box cut
    data_cut = raw_full_res_data_img[0, new_bounding_box[0]:new_bounding_box[1] + 1,
               new_bounding_box[2]:new_bounding_box[3] + 1, new_bounding_box[4]:new_bounding_box[5] + 1]
    label_cut = raw_full_res_label_img[new_bounding_box[0]:new_bounding_box[1] + 1,
                new_bounding_box[2]:new_bounding_box[3] + 1, new_bounding_box[4]:new_bounding_box[5] + 1]

    # data must have channel shape, (1, slices, x, y)
    data_cut = np.expand_dims(data_cut, axis=0)

    cut_sum = label_cut.sum()
    full_res_sum = raw_full_res_label_img.sum()
    print('debug, Does cut and original label contain the same amount of pixels?', cut_sum == full_res_sum, cut_sum,
          full_res_sum)
    # assert cut_sum == full_res_sum

    # debug
    if show_debug:
        print('debug bounding box sizes',
              get_bounding_box_3D_size(*bounding_box),
              get_bounding_box_3D_size(*new_bounding_box))
        print('debug bounding boxes', bounding_box, new_bounding_box)
        debug_preview_model_output(low_res_data_img, low_res_label_img, model_output_img)
        debug_preview_cuts(exp_model_output_img, new_bounding_box, data_cut, label_cut)

    return data_cut, label_cut, new_bounding_box

