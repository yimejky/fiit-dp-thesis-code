import numpy as np
import torch

from src.consts import DESIRE_BOUNDING_BOX_SIZE
from src.dataset.cut_dataset_debug_helpers import debug_preview_cuts, debug_preview_model_output
from src.dataset.dataset_transforms import get_norm_transform
from src.dataset.transform_input import transform_input

from src.helpers.bounding_box_helpers import get_bounding_box_3D_size, get_final_bounding_box_slice, get_bounding_box_3D
from src.helpers.outlier_helpers import get_img_outliers_pixels


def expand_image(input_img, expand_factor=16):  # input numpy shape (1, 1, MAX_PADDING_SLICES, x, x)
    expanded_input_img = np.repeat(np.repeat(input_img, expand_factor, axis=3), expand_factor, axis=4)
    return expanded_input_img


def insert_to_model(low_res_model,
                    low_res_device,
                    low_res_data_img,
                    low_res_mask_threshold,
                    show_debug=False):
    # getting low res segmentation
    exp_low_res_data_img_np = np.expand_dims(low_res_data_img, axis=0)
    exp_low_res_data_img = torch.from_numpy(exp_low_res_data_img_np).float().to(low_res_device)

    model_output_img = low_res_model(exp_low_res_data_img)
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
        low_res_device,
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
    model_output_img = insert_to_model(low_res_model,
                                       low_res_device,
                                       low_res_data_img,
                                       low_res_mask_threshold,
                                       show_debug=show_debug)
    positive_pixels_count = np.sum(model_output_img > 0)

    # outliers detection
    remove_pixel_idx = get_img_outliers_pixels(model_output_img[0, 0])
    for x in remove_pixel_idx:
        model_output_img[tuple([0, 0, *x])] = 0
    print(f'get_full_res_cut: Removing {len(remove_pixel_idx)}/{positive_pixels_count} outlier pixels')

    # expanding low res int mask to high res
    exp_model_output_img = expand_image(model_output_img, expand_factor=16)  # shape (1, 1, 160, 512, 512)

    # getting bounding box
    bounding_box = get_bounding_box_3D(exp_model_output_img[0][0]) # removing batch, channel dimension
    new_bounding_box = get_final_bounding_box_slice(bounding_box, desire_bounding_box_size)

    # getting bounding box cut
    data_cut = raw_full_res_data_img[
               :,
               new_bounding_box[0]:new_bounding_box[1] + 1,
               new_bounding_box[2]:new_bounding_box[3] + 1,
               new_bounding_box[4]:new_bounding_box[5] + 1]
    label_cut = raw_full_res_label_img[
                :,
                new_bounding_box[0]:new_bounding_box[1] + 1,
                new_bounding_box[2]:new_bounding_box[3] + 1,
                new_bounding_box[4]:new_bounding_box[5] + 1]

    cut_sum = label_cut.sum()
    full_res_sum = raw_full_res_label_img.sum()
    print('get_full_res_cut: Does cut and original label contain the same amount of pixels?',
          cut_sum == full_res_sum,
          cut_sum,
          full_res_sum)
    # assert cut_sum == full_res_sum

    # debug
    if show_debug:
        print('get_full_res_cut: Bounding box sizes',
              get_bounding_box_3D_size(*bounding_box),
              get_bounding_box_3D_size(*new_bounding_box))
        print('get_full_res_cut: Bounding boxes', bounding_box, new_bounding_box)
        debug_preview_model_output(low_res_data_img, low_res_label_img, model_output_img)
        debug_preview_cuts(exp_model_output_img, new_bounding_box, data_cut, label_cut)

    return data_cut, label_cut, new_bounding_box


def get_cut_lists(low_res_model,
                  low_res_device,
                  low_res_dataset,
                  full_res_dataset,
                  cut_full_res_dataset,
                  low_res_mask_threshold=0.5):
    for dataset_index in range(len(full_res_dataset)):
        print(f'get_cut_lists: Cutting index {dataset_index}')

        # getting raw data
        raw_low_res_data_img = low_res_dataset.data_list[dataset_index]
        raw_low_res_label_img = low_res_dataset.label_list[dataset_index]
        raw_full_res_data_img = full_res_dataset.data_list[dataset_index]
        raw_full_res_label_img = full_res_dataset.label_list[dataset_index]

        # parsing
        data_cut, label_cut, new_bounding_box = get_full_res_cut(low_res_model=low_res_model,
                                                                 low_res_device=low_res_device,
                                                                 raw_low_res_data_img=raw_low_res_data_img,
                                                                 raw_low_res_label_img=raw_low_res_label_img,
                                                                 raw_full_res_data_img=raw_full_res_data_img,
                                                                 raw_full_res_label_img=raw_full_res_label_img,
                                                                 low_res_mask_threshold=low_res_mask_threshold,
                                                                 desire_bounding_box_size=DESIRE_BOUNDING_BOX_SIZE,
                                                                 show_debug=False)

        # inserting to cut dataset
        cut_full_res_dataset.data_list[dataset_index] = data_cut
        cut_full_res_dataset.label_list[dataset_index] = label_cut

    return cut_full_res_dataset
