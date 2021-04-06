import SimpleITK as sitk
import numpy as np


def transform_sitk(fixed, moving, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    return resampler.Execute(moving)


def transform_full_np(fixed_np, moving_np, output_transform):
    fixed_data, fixed_label = fixed_np
    moving_data, moving_label = moving_np

    # parsing input to floats
    fixed_data = fixed_data.astype(np.float32)[0]

    # converting to sitk
    fixed_data_sitk = sitk.GetImageFromArray(fixed_data)
    moving_label_sitk = sitk.GetImageFromArray(moving_label)

    # use registration transform to transform atlas
    trans_fixed_label_sitk = transform_sitk(fixed_data_sitk, moving_label_sitk, output_transform)

    return trans_fixed_label_sitk
