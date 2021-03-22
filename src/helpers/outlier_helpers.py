import numpy as np


def get_img_outliers_pixels(input_img):
    pixels_cords = np.argwhere(input_img > 0)
    q25, q75 = np.percentile(pixels_cords, 25, axis=0, keepdims=True)[0], \
               np.percentile(pixels_cords, 75, axis=0, keepdims=True)[0]
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    # print('debug', lower, upper)

    tmp_outlier_lower = (pixels_cords[:, 0] < lower[0]) | (pixels_cords[:, 1] < lower[1]) | (
            pixels_cords[:, 2] < lower[2])
    tmp_outlier_upper = (pixels_cords[:, 0] > upper[0]) | (pixels_cords[:, 1] > upper[1]) | (
            pixels_cords[:, 2] > upper[2])

    return pixels_cords[tmp_outlier_lower | tmp_outlier_upper]
