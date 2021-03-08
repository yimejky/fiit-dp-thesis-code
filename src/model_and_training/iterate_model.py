import logging

import numpy as np
import torch

from contextlib import nullcontext

from src.model_and_training.loss_batch import loss_batch


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

            inputs, labels = data
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

        num_sums = np.sum(nums)
        final_loss = np.sum(np.multiply(losses, nums)) / num_sums
        final_dice = np.sum(np.multiply(dices, nums)) / num_sums
        return final_loss, final_dice
