import SimpleITK as sitk

from src.helpers.registration_helpers.preview_reg_transform import preview_reg_transform


def get_regis_trans_non_rigid_sitk(fixed,
                                   moving,
                                   numberOfIterations=100,
                                   show=True,
                                   show_eval=True,
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

    # print('Debug', fixed.GetPixelIDTypeAsString(), moving.GetPixelIDTypeAsString())
    output_transform = R.Execute(fixed, moving)

    if show_eval:
        print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        print(f"  Iteration: {R.GetOptimizerIteration()}")
        print(f"  Metric value: {R.GetMetricValue()}")

    if preview:
        preview_reg_transform(fixed, moving, output_transform, figsize=figsize)

    return output_transform
