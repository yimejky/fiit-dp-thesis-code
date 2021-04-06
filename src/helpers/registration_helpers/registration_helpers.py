import SimpleITK as sitk
import numpy as np

from src.helpers.registration_helpers.get_regis_trans_rigid_sitk import get_regis_trans_rigid_sitk
from src.helpers.registration_helpers.transform import transform_full_np


def get_regis_trans(fixed_np,
                    moving_data_sitk,
                    numberOfIterations=500,
                    show=False,
                    show_eval=True,
                    preview=False,
                    figsize=(16, 16),
                    get_reg_trans_sitk=get_regis_trans_rigid_sitk):
    fixed_data, fixed_label = fixed_np
    fixed_data = fixed_data.astype(np.float32)[0]
    fixed_data_sitk = sitk.GetImageFromArray(fixed_data)

    # get registration
    output_transform = get_reg_trans_sitk(fixed_data_sitk,
                                          moving_data_sitk,
                                          numberOfIterations=numberOfIterations,
                                          show=show,
                                          show_eval=show_eval,
                                          preview=preview,
                                          figsize=figsize)

    return output_transform


def get_transformed_label_np(fixed_np,
                             moving_np,
                             numberOfIterations=500,
                             show=False,
                             show_eval=True,
                             preview=False,
                             figsize=(16, 16),
                             get_reg_trans_sitk=get_regis_trans_rigid_sitk):
    moving_data, moving_label = moving_np
    moving_data = moving_data.astype(np.float32)
    moving_data_sitk = sitk.GetImageFromArray(moving_data)

    # get registration
    output_transform = get_regis_trans(fixed_np,
                                       moving_data_sitk,
                                       numberOfIterations=numberOfIterations,
                                       show=show,
                                       show_eval=show_eval,
                                       preview=preview,
                                       figsize=figsize,
                                       get_reg_trans_sitk=get_reg_trans_sitk)

    trans_fixed_label_sitk = transform_full_np(fixed_np, moving_np, output_transform)
    trans_fixed_label_np = sitk.GetArrayFromImage(trans_fixed_label_sitk)

    return trans_fixed_label_np


def create_regis_trans_list(dataset,
                            atlas_input_data_np,
                            numberOfIterations=500,
                            get_reg_trans_sitk=get_regis_trans_rigid_sitk):
    merged_list = list()

    for dataset_index in range(len(dataset)):
        dataset_input = dataset.get_raw_item_with_label_filter(dataset_index)

        atlas_input_data_np = atlas_input_data_np.astype(np.float32)
        atlas_input_data_sitk = sitk.GetImageFromArray(atlas_input_data_np)

        reg_trans = get_regis_trans(dataset_input,
                                    atlas_input_data_sitk,
                                    numberOfIterations=numberOfIterations,
                                    show=False,
                                    show_eval=True,
                                    preview=False,
                                    get_reg_trans_sitk=get_reg_trans_sitk)

        print(f'Registration done for index: {dataset_index}')
        merged_list.append(reg_trans)
    return merged_list


def create_regis_list(dataset,
                      atlas_input,
                      numberOfIterations=500,
                      get_reg_trans_sitk=get_regis_trans_rigid_sitk):
    merged_list = list()

    for dataset_index in range(len(dataset)):
        dataset_input = dataset.get_raw_item_with_label_filter(dataset_index)

        reg_output = get_transformed_label_np(dataset_input,
                                              atlas_input,
                                              numberOfIterations=numberOfIterations,
                                              show=False,
                                              show_eval=True,
                                              preview=False,
                                              get_reg_trans_sitk=get_reg_trans_sitk)
        reg_output = reg_output.astype(np.float32)
        print(f'Registration done for index: {dataset_index}')
        merged_output = np.array([dataset.data_list[dataset_index][0], reg_output])

        merged_list.append(merged_output)
    return merged_list


def trans_list(dataset, atlas_input_np, regis_trans_list):
    registration_list = list()

    for dataset_index in range(len(dataset)):
        dataset_input = dataset.get_raw_item_with_label_filter(dataset_index)

        reg_output_sitk = transform_full_np(dataset_input, atlas_input_np, regis_trans_list[dataset_index])

        reg_output_np = sitk.GetArrayFromImage(reg_output_sitk)
        reg_output_np = reg_output_np.astype(np.float32)
        merged_output = np.array([dataset.data_list[dataset_index][0], reg_output_np])

        registration_list.append(merged_output)
        print(f'Transform done for index: {dataset_index}')

    return registration_list

