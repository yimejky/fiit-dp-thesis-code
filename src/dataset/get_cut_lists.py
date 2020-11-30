from src.consts import DESIRE_BOUNDING_BOX_SIZE
from src.dataset.get_full_res_cut import get_full_res_cut


def get_cut_lists(low_res_model,
                  low_res_device,
                  low_res_dataset,
                  full_res_dataset,
                  cut_full_res_dataset,
                  low_res_mask_threshold=0.5):
    for dataset_index in range(len(full_res_dataset)):
        print(f'getting cut index {dataset_index}')

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

