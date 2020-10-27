import matplotlib.pyplot as plt

from IPython.display import display
from ipywidgets import widgets


def preview_dataset(dataset, preview_index=0, show_hist=False):
    data, label = dataset[preview_index]
    max_slices = label.shape[0]

    print(f'data max {data.max()}, min {data.min()}')
    print(f'label max {label.max()}, min {label.min()}')

    def f(slice_index):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(data[0][slice_index], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(label[slice_index])
        plt.show()

        if show_hist:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.hist(data.flatten(), 128)
            plt.subplot(1, 2, 2)
            plt.hist(label.flatten(), 128)
            plt.show()

    sliceSlider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=(max_slices - 1) / 2)
    ui = widgets.VBox([widgets.HBox([sliceSlider])])
    out = widgets.interactive_output(f, {'slice_index': sliceSlider})
    # noinspection PyTypeChecker
    display(ui, out)
