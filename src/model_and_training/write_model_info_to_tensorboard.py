from operator import itemgetter

from src.model_and_training.getters.get_loaders import get_loaders


def get_indices(dataset):
    return str(sorted([i + 1 for i in dataset.indices]))


def write_model_info_to_tensorboard(model_info,  train_dataset, valid_dataset, test_dataset):
    tensorboard_writer = model_info['tensorboard_writer']

    model, model_params = itemgetter('model', 'model_params')(model_info)
    epochs, learning_rate, train_batch_size = itemgetter('epochs', 'learning_rate', 'train_batch_size')(model_info)
    optimizer, criterion = itemgetter('optimizer', 'criterion')(model_info)
    model_total_params, model_total_trainable_params = itemgetter(
        'model_total_params', 'model_total_trainable_params')(model_info)

    train_dataloader, valid_dataloader, test_dataloader = get_loaders(train_batch_size,
                                                                      train_dataset, valid_dataset, test_dataset)
    images, labels = iter(train_dataloader).next()
    tensorboard_writer.add_graph(model, images.to(model_info['device']))

    tensorboard_writer.add_text('model', str(type(model).__name__))
    tensorboard_writer.add_text('model_params', str(model_params))
    tensorboard_writer.add_text('optimizer', str(type(optimizer).__name__))
    tensorboard_writer.add_text('criterion', str(type(criterion).__name__))
    tensorboard_writer.add_text('epochs', str(epochs))
    tensorboard_writer.add_text('learning_rate', str(learning_rate))
    tensorboard_writer.add_text('train_batch_size', str(train_batch_size))
    tensorboard_writer.add_text('train_dataset_indices', get_indices(train_dataset))
    tensorboard_writer.add_text('valid_dataset_indices', get_indices(valid_dataset))
    tensorboard_writer.add_text('test_dataset_indices', get_indices(test_dataset))
    tensorboard_writer.add_text('model_number_of_params', str(model_total_params))
    tensorboard_writer.add_text('model_number_of_trainable_params', str(model_total_trainable_params))

    return tensorboard_writer
