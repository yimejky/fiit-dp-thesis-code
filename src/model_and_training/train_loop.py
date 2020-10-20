from datetime import timedelta
from time import time

from src.model_and_training.iterate_model import iterate_model
from src.model_and_training.training_helpers import checkpoint_model


def train_loop(model, model_name, optimizer, criterion, epochs, device,
               tensorboard_writer,
               train_dataloader, valid_dataloader, test_dataloader):
    print('Running training loop')
    start_time = last_time = time()

    for epoch_i in range(epochs):
        train_loss, train_dsc = iterate_model(train_dataloader, model, optimizer, criterion, device, is_eval=False)
        print('Epoch [%d] train done' % (epoch_i + 1))
        valid_loss, valid_dsc = iterate_model(valid_dataloader, model, optimizer, criterion, device, is_eval=True)
        print('Epoch [%d] valid done' % (epoch_i + 1))

        delta_start_time = time() - start_time
        delta_last_time = time() - last_time
        print_epochs = (epoch_i + 1, delta_start_time, delta_last_time, train_loss, valid_loss, train_dsc, valid_dsc)
        print(
            'Epoch [%d] T %.2fs, deltaT %.2fs, loss: train %.5f, valid %.5f, dsc: train %.5f, valid %.5f' % print_epochs)
        last_time = time()

        checkpoint_model(model_name, epoch_i + 1, model, optimizer)
        tensorboard_step_value = (epoch_i + 1) * len(train_dataloader) * train_dataloader.batch_size
        tensorboard_writer.add_scalars('loss', {"train": train_loss, "valid": valid_loss}, tensorboard_step_value)
        tensorboard_writer.add_scalars('dsc', {"train": train_dsc, "valid": valid_dsc}, tensorboard_step_value)

    # TODO prevent using test
    # test_loss, test_dsc = iterate_model(test_dataloader, model, optimizer, criterion, device, is_eval=True)
    # print('test: loss %.4f, dsc %.4f' % (test_loss, test_dsc))
    elapsed_time_training = time() - start_time
    delta_elapsed_time_training = timedelta(seconds=elapsed_time_training)
    print_elapsed = ':'.join(str(delta_elapsed_time_training).split(".")[:1])
    print(f'Elapsed time {print_elapsed}')
