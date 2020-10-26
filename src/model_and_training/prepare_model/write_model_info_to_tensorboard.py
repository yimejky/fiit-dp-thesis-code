from operator import itemgetter


def get_indices(dataset):
    return str(sorted([i + 1 for i in dataset.indices]))


def write_model_info_to_tensorboard(model_info,  train_dataset, valid_dataset, test_dataset):
    tensorboard_writer = model_info['tensorboard_writer']

    learning_rate, epochs = itemgetter('learning_rate', 'epochs')(model_info)
    criterion = model_info['criterion']
    model_total_params, model_total_trainable_params = itemgetter(
        'model_total_params', 'model_total_trainable_params')(model_info)

    tensorboard_writer.add_text('data_indices_train', get_indices(train_dataset))
    tensorboard_writer.add_text('data_indices_valid', get_indices(valid_dataset))
    tensorboard_writer.add_text('data_indices_test', get_indices(test_dataset))
    tensorboard_writer.add_text('optimizer_learning_rate', str(learning_rate))
    tensorboard_writer.add_text('epochs', str(epochs))
    tensorboard_writer.add_text('loss_function', str(type(criterion).__name__))
    tensorboard_writer.add_text('model_number_of_params', str(model_total_params))
    tensorboard_writer.add_text('model_number_of_trainable_params', str(model_total_trainable_params))

    return tensorboard_writer
