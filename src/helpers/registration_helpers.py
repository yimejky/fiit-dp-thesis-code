import SimpleITK as sitk
import numpy as np

from src.helpers import preview_3d_image


def get_registration_transform_sitk(fixed, moving, numberOfIterations=500, show=True, preview=True, figsize=(16, 16)):
    """https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod3_docs.html"""

    def command_iteration(method):
        if show:
            if method.GetOptimizerIteration() == 0:
                print("Estimated Scales: ", method.GetOptimizerScales())
            print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                                  method.GetMetricValue(),
                                                  method.GetOptimizerPosition()))

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                               minStep=1e-4,
                                               numberOfIterations=numberOfIterations,
                                               gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromIndexShift()
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity3DTransform())
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    output_transform = R.Execute(fixed, moving)

    if show:
        print("-------")
        print(output_transform)
        print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

    if preview:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(1)
        resampler.SetTransform(output_transform.GetInverse())

        out = resampler.Execute(fixed)

        simg1 = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkUInt8)
        simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
        preview_3d_image(cimg, figsize=figsize)

    return output_transform


def transform_sitk(fixed, moving, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    return resampler.Execute(fixed)


def get_registration_transform_np(fixed_np, moving_np,
                                  numberOfIterations=500,
                                  show=False,
                                  preview=False,
                                  figsize=(16, 16)):
    fixed_data, fixed_label = fixed_np
    moving_data, moving_label = moving_np

    # parsing input to floats
    fixed_data = fixed_data.astype(np.float32)
    moving_data = moving_data.astype(np.float32)[0]
    if show:
        print(fixed_data.dtype, moving_data.dtype, fixed_data.shape, moving_data.shape)

    # converting to sitk
    fixed_data_sitk = sitk.GetImageFromArray(fixed_data)
    fixed_label_sitk = sitk.GetImageFromArray(fixed_label)
    moving_data_sitk = sitk.GetImageFromArray(moving_data)

    # get registration
    output_transform = get_registration_transform_sitk(fixed_data_sitk, moving_data_sitk,
                                                       numberOfIterations=numberOfIterations,
                                                       show=show,
                                                       preview=preview,
                                                       figsize=figsize)

    # use registration transform to transform atlas
    trans_fixed_label = transform_sitk(fixed_label_sitk, moving_data_sitk, output_transform.GetInverse())
    trans_fixed_label_np = sitk.GetArrayFromImage(trans_fixed_label)

    return trans_fixed_label_np

