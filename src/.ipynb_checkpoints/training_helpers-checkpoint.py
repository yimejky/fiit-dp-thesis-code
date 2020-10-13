import numpy as np
import torch

from contextlib import nullcontext
from pathlib import Path

from torchsummary import summary

from src.consts import MAX_PADDING_SLICES, IN_COLAB
from src.helpers.calc_dsc import calc_dsc


def loss_batch(model, optimizer, loss_func, xb, yb, calc_backward=False):
    """ source https://pytorch.org/tutorials/beginner/nn_tutorial.html """
    prediction = model(xb)
    
    dsc = calc_dsc(yb, prediction)
    loss = loss_func(prediction[:, 0], yb)

    if calc_backward:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), dsc, len(xb)


def iterate_model(dataloader, model, optimizer, loss_func, device, is_eval=False):
    if is_eval:
        info_text = 'eval'
        model.eval()
        grad_context = torch.no_grad()
    else:
        info_text = 'train'
        model.train()
        grad_context = nullcontext()

    with grad_context:
        losses, dices, nums = [], [], []
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            item_loss, item_dsc, inputs_len = loss_batch(model, optimizer, loss_func, inputs, labels, not is_eval)
            losses.append(item_loss)
            dices.append(item_dsc)
            nums.append(inputs_len)
            if is_eval:
                print(f'Batch {info_text} [%i] loss %.5f, dsc %.5f' % (i+1, item_loss, item_dsc))

        num_sums = np.sum(nums)
        final_loss = np.sum(np.multiply(losses, nums)) / num_sums
        final_dice = np.sum(np.multiply(dices, nums)) / num_sums
        return final_loss, final_dice


def checkpoint_model(model_name, epoch, model, optimizer):
    folder_path = f"models/{model_name}"
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, f'{folder_path}/checkpoint_{model_name}_epoch_{epoch}.pkl')


def show_model_info(model):
    if (IN_COLAB):
        summary(model, (1, MAX_PADDING_SLICES, 128, 128), batch_size=1)

    unet_total_params = sum(p.numel() for p in model.parameters())
    unet_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model number of params: {unet_total_params}, trainable {unet_total_params_trainable}')


