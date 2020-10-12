import numpy as np


def get_bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    slice_min, slice_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return slice_min, slice_max, x_min, x_max


def get_bounding_box_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    slice_min, slice_max = np.where(r)[0][[0, -1]]
    x_min, x_max = np.where(c)[0][[0, -1]]
    y_min, y_max = np.where(z)[0][[0, -1]]

    return slice_min, slice_max, x_min, x_max, y_min, y_max


def get_bounding_box_3D_size(slice_min, slice_max, x_min, x_max, y_min, y_max):
    return slice_max - slice_min + 1, x_max - x_min + 1, y_max - y_min + 1


def get_final_bounding_box_slice(bounding_box, desire_box_size):
    box_size = get_bounding_box_3D_size(*bounding_box)

    delta_box = desire_box_size - np.array(box_size)
    left_delta_box = delta_box // 2
    right_delta_box = delta_box - left_delta_box
    print('debug box delta', delta_box)
    # print('debug box delta', 'left', left_delta_box, 'right', right_delta_box)

    slice_a = bounding_box[0] - left_delta_box[0]
    slice_b = bounding_box[1] + right_delta_box[0]
    x_a = bounding_box[2] - left_delta_box[1]
    x_b = bounding_box[3] + right_delta_box[1]
    y_a = bounding_box[4] - left_delta_box[2]
    y_b = bounding_box[5] + right_delta_box[2]

    new_bounding_box = slice_a, slice_b, x_a, x_b, y_a, y_b

    return new_bounding_box


def get_closest_number(target_num, pooling_layers=3, offset=0):
    i = 1
    start_num = 2 ** pooling_layers
    found_num = start_num * i
    while found_num < target_num + offset:
        i += 1
        found_num = start_num * i
    return found_num


def get_dividable_bounding_box(bounding_box_sizes, pooling_layers=3, offset=12):
    return [get_closest_number(x, pooling_layers, offset) for x in bounding_box_sizes]


if __name__ == "__main__":
    tmp_test = np.zeros((7, 5, 5))
    tmp_test[3, 3, 3] = 1
    tmp_test[4, 3, 3] = 1

    tmp_box = get_bounding_box_3D(tmp_test)
    tmp_box_size = get_bounding_box_3D_size(*tmp_box)
    tmp_final_box = get_final_bounding_box_slice(tmp_box, np.array([4, 3, 3]))
    tmp_final_box_size = get_bounding_box_3D_size(*tmp_final_box)

    print('box first, last pixels indx', tmp_box)
    print('box size', tmp_box_size)
    print('final box first, last pixels idx', tmp_final_box)
    print('final box size', tmp_final_box_size)

    tmp_cut = tmp_test[tmp_final_box[0]:tmp_final_box[1]+1, tmp_final_box[2]:tmp_final_box[3]+1, tmp_final_box[4]:tmp_final_box[5]+1]
    print('final cut', tmp_cut.shape, tmp_cut)


