import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display
from ipywidgets import widgets


def preview_3d_image(img, figsize=(16, 16), show_slice=None, show_unique=False, minmax=False):
    if type(img) is sitk.SimpleITK.Image:
        img = sitk.GetArrayFromImage(img)

    max_slices = img.shape[0]

    def f(slice_index):
        plt.figure(figsize=figsize)
        if minmax:
            plt.imshow(img[slice_index], vmin=np.min(img), vmax=np.max(img))
        else:
            plt.imshow(img[slice_index])
        plt.show()

        if show_unique:
            print(f"debug: {img.min()}, {img.max()}")
            print(f"debug: unique {np.unique(img[slice_index])}")

    if show_slice:
        sliceSlider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=show_slice)
    else:
        sliceSlider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=(max_slices - 1) / 2)

    ui = widgets.VBox([widgets.HBox([sliceSlider])])
    out = widgets.interactive_output(f, {'slice_index': sliceSlider})
    # noinspection PyTypeChecker
    display(ui, out)
