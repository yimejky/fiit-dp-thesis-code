from src.helpers.calc_dsc import calc_dsc


def loss_batch(model, optimizer, loss_func, model_input, true_output, calc_backward=False):
    """ source https://pytorch.org/tutorials/beginner/nn_tutorial.html """
    prediction = model(model_input)

    dsc = calc_dsc(true_output, prediction)
    loss = loss_func(prediction[:, 0], true_output)

    if calc_backward:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), dsc, len(model_input)
