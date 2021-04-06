import SimpleITK as sitk

from src.helpers.registration_helpers.preview_reg_transform import preview_reg_transform


def get_regis_trans_rigid_sitk(fixed,
                               moving,
                               numberOfIterations=500,
                               show=True,
                               show_eval=True,
                               preview=True,
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
    # print(f"debug {fixed.GetPixelIDTypeAsString()}, {moving.GetPixelIDTypeAsString()}")
    # print(f"debug {fixed.GetDimension()}, {moving.GetDimension()}")

    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity3DTransform())
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    output_transform = R.Execute(fixed, moving)

    if show_eval:
        print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        print(f"  Iteration: {R.GetOptimizerIteration()}")
        print(f"  Metric value: {R.GetMetricValue()}")

    if preview:
        preview_reg_transform(fixed, moving, output_transform, figsize=figsize)

    return output_transform
