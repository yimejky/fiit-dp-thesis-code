import matplotlib.pyplot as plt

from IPython.display import display
from ipywidgets import widgets
from src.consts import MAX_PADDING_SLICES


def preview_datasets(datasets_obj, preview_index=2, show_hist=False):
    aSlider = widgets.IntSlider(min=0, max=MAX_PADDING_SLICES-1, step=1, value=101)
    ui = widgets.VBox([widgets.HBox([aSlider])])
    data, label = datasets_obj["dataset"][preview_index]

    print(f'data max {data.max()}, min {data.min()}')
    print(f'label max {label.max()}, min {label.min()}')

    def f(a):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(data[0][a], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(label[a])
        plt.show()

        if show_hist:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.hist(data.flatten(), 128)
            plt.subplot(1, 2, 2)
            plt.hist(label.flatten(), 128)
            plt.show()

    out = widgets.interactive_output(f, {'a': aSlider})
    display(ui, out)