import matplotlib.pyplot as plt

from IPython.display import display
from ipywidgets import widgets

from src.dataset.dataset_transforms import get_dataset_transform
from src.dataset.transform_input import transform_input


def preview_dataset(dataset, preview_index=0, show_hist=False, use_transform=False):
    data, label = dataset.get_raw_item_with_label_filter(preview_index)  # equivalent dataset[preview_index]
    if use_transform:
        transform = get_dataset_transform()
        data, label = transform_input(data, label, transform)

    max_channels = label.shape[0]
    max_slices = label.shape[1]

    print(f'data max {data.max()}, min {data.min()}')
    print(f'label max {label.max()}, min {label.min()}')

    def f(slice_index, label_channel):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(data[0, slice_index], cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(label[label_channel, slice_index])
        plt.show()

        if show_hist:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.hist(data.flatten(), 128)
            plt.subplot(1, 2, 2)
            plt.hist(label.flatten(), 128)
            plt.show()

    sliceSlider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=(max_slices - 1) / 2)
    labelChannelSlider = widgets.IntSlider(min=0, max=max_channels - 1, step=1, value=(max_channels - 1) / 2)
    ui = widgets.VBox([widgets.HBox([sliceSlider, labelChannelSlider])])
    out = widgets.interactive_output(f, {'slice_index': sliceSlider, 'label_channel': labelChannelSlider})
    # noinspection PyTypeChecker
    display(ui, out)
