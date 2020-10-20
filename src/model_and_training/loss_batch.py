import logging

from src.helpers.calc_dsc import calc_dsc


def loss_batch(model, optimizer, loss_func, model_input, true_output, calc_backward=False):
    """ source https://pytorch.org/tutorials/beginner/nn_tutorial.html """
    prediction = model(model_input)
    # model input has channel dimension, our implementation of label dont
    prediction = prediction[:, 0]

    dsc = calc_dsc(true_output, prediction)
    loss = loss_func(prediction, true_output)

    logging.debug(f'loss_batch1 true_output {true_output.shape}, pred {prediction.shape}')
    logging.debug(f'loss_batch2 dsc {dsc}, loss {loss.item()}, model_input len {len(model_input)}')

    if calc_backward:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), dsc, len(model_input)
