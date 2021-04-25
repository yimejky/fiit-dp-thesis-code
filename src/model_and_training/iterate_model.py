import logging

import numpy as np
import torch

from contextlib import nullcontext

from src.dataset.dataset_transforms import get_norm_transform, get_dataset_transform
from src.dataset.transform_input import transform_input
from src.model_and_training.loss_batch import loss_batch

norm_trans = get_norm_transform()
dataset_trans = get_dataset_transform()


def iterate_model(dataloader, model, optimizer, loss_func, device, is_eval=False):
    model.is_train = not is_eval

    if is_eval:
        info_text = 'eval'
        model.eval()
        grad_context = torch.no_grad()
    else:
        info_text = 'train'
        model.train()
        optimizer.zero_grad()
        grad_context = nullcontext()

    with grad_context:
        losses, dices, nums = [], [], []
        for i, data in enumerate(dataloader):
            model.actual_step = i

            # torch.io data augmentation and transforms
            inputs, labels = data
            transform = norm_trans if is_eval else dataset_trans

            for i in range(inputs.shape[0]):
                tmp_inputs, tmp_labels = transform_input(inputs[i], labels[i], transform)
                log_msg = f'iterate_model_v3v1_1: {tmp_inputs.shape}, {tmp_labels.shape}'
                # print(log_msg)
                logging.debug(log_msg)
                inputs[i] = torch.tensor(tmp_inputs)
                labels[i] = torch.tensor(tmp_labels)

            # converting to float
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            item_loss, item_dsc, inputs_len = loss_batch(model, optimizer, loss_func, inputs, labels,
                                                         calc_backward=not is_eval)
            losses.append(item_loss)
            dices.append(item_dsc)
            nums.append(inputs_len)

            # batch done
            msg = f'Batch {info_text} [%i] loss %.5f, dsc %.5f' % (i + 1, item_loss, item_dsc)
            print(msg)
            logging.debug(f'iterate_model0 optimizer {optimizer.param_groups[0]["lr"]}')
            logging.debug(f'iterate_model1 batch info {msg}')

            # clearing
            del data
            del inputs
            del labels
            del item_loss
            del item_dsc
            torch.cuda.empty_cache()

        model.tensorboard_writer.add_text('epoch_items_dsc', str(dices), model.actual_epoch)
        model.tensorboard_writer.add_text('epoch_items_nums', str(nums), model.actual_epoch)
        model.tensorboard_writer.add_text('epoch_items_dataloaders', str(dataloader.dataset.indices), model.actual_epoch)

        num_sums = np.sum(nums)
        final_loss = np.sum(np.multiply(losses, nums)) / num_sums
        final_dice = np.sum(np.multiply(dices, nums)) / num_sums
        return final_loss, final_dice
