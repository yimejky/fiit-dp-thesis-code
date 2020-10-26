from operator import itemgetter

import torch
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import widgets
from IPython.display import display
from src.consts import MAX_PADDING_SLICES
from src.model_and_training.loss_batch import loss_batch


def show_model_dataset_pred_preview(model_info,
                                    dataset,
                                    dataset_index=None,
                                    figfile=None,
                                    max_slices=MAX_PADDING_SLICES,
                                    default_slice=(MAX_PADDING_SLICES - 1) // 2):
    model, device, optimizer, criterion = itemgetter('model', 'device', 'optimizer', 'criterion')(model_info)

    with torch.no_grad():
        if dataset_index is None:
            dataset_index = dataset.indices[0]

        model.eval()
        torch.cuda.empty_cache()
        print(f'showing number {dataset_index}')
        inputs, labels = dataset[dataset_index]
        inputs = torch.from_numpy(np.array([inputs])).to(device).float()
        labels = torch.from_numpy(np.array([labels])).to(device).float()
        prediction = model(inputs)

        item_loss, item_dsc, inputs_len = loss_batch(model, optimizer, criterion, inputs, labels)
        print(f'loss {item_loss}, dsc {item_dsc}, inputs_len {inputs_len}')

        inputs = inputs.cpu()
        labels = labels.cpu()
        prediction_np = prediction.cpu().detach().numpy()

        plt.hist(prediction_np[prediction_np > 0.01].flatten(), 20)
        plt.title('Distribution of prediction values')
        plt.show()

        def f(slice_index):
            plt.figure(figsize=(30, 16))
            tmp_ax = plt.subplot(1, 3, 1)
            tmp_ax.title.set_text('Input')
            plt.imshow(inputs[0, 0, slice_index], cmap="gray")

            tmp_ax = plt.subplot(1, 3, 2)
            tmp_ax.title.set_text('Label')
            plt.imshow(labels[0, slice_index], cmap="gray")

            tmp_ax = plt.subplot(1, 3, 3)
            tmp_ax.title.set_text('Prediction')
            plt.imshow(prediction_np[0, 0, slice_index], cmap="gray", vmin=0, vmax=1)
            #     plt.subplot(2, 2, 4)
            #     plt.imshow(prediction_np[0, 0, a], cmap="gray")

            if figfile is not None:
                plt.savefig(figfile, dpi=96)
            plt.show()

        aSlider = widgets.IntSlider(min=0, max=max_slices - 1, step=1, value=default_slice)
        ui = widgets.VBox([widgets.HBox([aSlider])])
        out = widgets.interactive_output(f, {'slice_index': aSlider})
        display(ui, out)

        print('DEBUG shapes', prediction[:, 0].shape, labels.shape, inputs.shape)
        print(f'DEBUG prediction max {prediction.max().item()}, min {prediction.min().item()}')
        print('DEBUG intersection', (labels.cpu() * prediction.cpu()[:, 0]).sum().item())
        print('DEBUG label sum', labels.cpu().sum().item())
        print('DEBUG prediction sum', prediction.cpu()[:, 0].sum().item())

        smooth = 1e-6
        y_pred = prediction.cpu()[:, 0].contiguous().view(-1)
        y_true = labels.cpu()[:].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + smooth) / (
                y_pred.sum() + y_true.sum() + smooth
        )

        print('DEBUG intersection2', intersection.item())
        print('DEBUG dsc', dsc.item())
        print('DEBUG MSE', (labels.cpu() - prediction.cpu()[:, 0]).pow(2).mean().item())
