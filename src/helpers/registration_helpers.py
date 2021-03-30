import SimpleITK as sitk
import numpy as np

from src.helpers.preview_3d_image import preview_3d_image


def preview_reg_transform(fixed, moving, output_transform, figsize=(16, 16)):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(1)
    resampler.SetTransform(output_transform)

    out = resampler.Execute(moving)

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    preview_3d_image(cimg, figsize=figsize)


def get_registration_transform_non_rigid_sitk(fixed, moving,
                                              numberOfIterations=100,
                                              show=True,
                                              preview=True,
                                              figsize=(16, 16)):
    """https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethodBSpline1_docs.html"""

    def command_iteration(method, bspline_transform):
        if show:
            if method.GetOptimizerIteration() == 0:
                # The BSpline is resized before the first optimizer
                # iteration is completed per level. Print the transform object
                # to show the adapted BSpline transform.
                print(bspline_transform)

        print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")

    def command_multi_iteration(method):
        if show:
            # The sitkMultiResolutionIterationEvent occurs before the
            # resolution of the transform. This event is used here to print
            # the status of the optimizer from the previous registration level.
            if method.GetCurrentLevel() > 0:
                print(f"Optimizer stop condition: {method.GetOptimizerStopConditionDescription()}")
                print(f" Iteration: {method.GetOptimizerIteration()}")
                print(f" Metric value: {method.GetMetricValue()}")

            print("--------- Resolution Changing ---------")

    # transform init
    transformDomainMeshSize = [2] * fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

    # register method
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation()

    lr = 5.0
    R.SetOptimizerAsGradientDescentLineSearch(lr,
                                              numberOfIterations,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)

    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=[1, 2, 5])
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([4, 2, 1])

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R, tx))
    R.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(R))

    print(fixed.GetPixelIDTypeAsString(), moving.GetPixelIDTypeAsString())
    output_transform = R.Execute(fixed, moving)

    if show:
        print("-------")
        print(tx)
        print(output_transform)
        print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        print(f" Iteration: {R.GetOptimizerIteration()}")
        print(f" Metric value: {R.GetMetricValue()}")

    if preview:
        preview_reg_transform(fixed, moving, output_transform, figsize=figsize)

    return output_transform


def get_registration_transform_rigid_sitk(fixed, moving, numberOfIterations=500, show=True, preview=True,
                                          figsize=(16, 16)):
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
        preview_reg_transform(fixed, moving, output_transform, figsize=figsize)

    return output_transform


def transform_sitk(fixed, moving, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    return resampler.Execute(moving)


def get_registration_transform_np(fixed_np, moving_np,
                                  numberOfIterations=500,
                                  show=False,
                                  preview=False,
                                  figsize=(16, 16),
                                  get_reg_trans_sitk=get_registration_transform_rigid_sitk):
    fixed_data, fixed_label = fixed_np
    moving_data, moving_label = moving_np

    # parsing input to floats
    fixed_data = fixed_data.astype(np.float32)[0]
    moving_data = moving_data.astype(np.float32)
    if show:
        print(fixed_data.dtype, moving_data.dtype, fixed_data.shape, moving_data.shape)

    # converting to sitk
    fixed_data_sitk = sitk.GetImageFromArray(fixed_data)
    moving_data_sitk = sitk.GetImageFromArray(moving_data)
    moving_label_sitk = sitk.GetImageFromArray(moving_label)

    # get registration
    output_transform = get_reg_trans_sitk(fixed_data_sitk,
                                          moving_data_sitk,
                                          numberOfIterations=numberOfIterations,
                                          show=show,
                                          preview=preview,
                                          figsize=figsize)

    # use registration transform to transform atlas
    trans_fixed_label = transform_sitk(fixed_data_sitk, moving_label_sitk, output_transform)
    trans_fixed_label_np = sitk.GetArrayFromImage(trans_fixed_label)

    return trans_fixed_label_np


def create_registration_list(dataset,
                             atlas_input,
                             numberOfIterations=500,
                             get_reg_trans_sitk=get_registration_transform_rigid_sitk):
    merged_list = list()

    for dataset_index in range(len(dataset)):
        dataset_input = dataset.get_raw_item_with_label_filter(dataset_index)

        reg_output = get_registration_transform_np(dataset_input,
                                                   atlas_input,
                                                   numberOfIterations=numberOfIterations,
                                                   show=False,
                                                   preview=False,
                                                   get_reg_trans_sitk=get_reg_trans_sitk)
        reg_output = reg_output.astype(np.float32)
        print(f'Registration done for index: {dataset_index}')
        merged_output = np.array([dataset.data_list[dataset_index][0], reg_output])

        merged_list.append(merged_output)
    return merged_list
