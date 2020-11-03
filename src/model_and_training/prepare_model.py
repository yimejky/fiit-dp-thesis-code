import datetime

from src.model_and_training import ModelInfo
from src.model_and_training.getters.get_criterion import get_criterion
from src.model_and_training.getters.get_device import get_device
from src.model_and_training.getters.get_loaders import get_loaders
from src.model_and_training.getters.get_model import get_model
from src.model_and_training.getters.get_model_params import get_model_params
from src.model_and_training.getters.get_optimizer import get_optimizer
from src.model_and_training.getters.get_tensorboard_writer import get_tensorboard_writer
from src.model_and_training.write_model_info_to_tensorboard import write_model_info_to_tensorboard


def prepare_model(epochs=30,  # 30 x train_size = number of steps
                  learning_rate=5e-4,
                  in_channels=16,
                  dropout_rate=0.2,
                  train_batch_size=1,
                  model_name=None,
                  train_dataset=None,
                  valid_dataset=None,
                  test_dataset=None):
    # Params
    log_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if model_name is None:
        model_name = f'{log_date}_3d_unet'

    # Setting up model and stuff
    device = get_device()
    criterion = get_criterion()

    # Architecture and optimizer with state
    model_params = {
        "in_channels": in_channels,
        "dropout_rate": dropout_rate,
    }
    model = get_model(device, model_params)
    optimizer = get_optimizer(model=model, learning_rate=learning_rate)

    # Data loaders
    train_dataloader, valid_dataloader, test_dataloader = get_loaders(train_batch_size,
                                                                      train_dataset, valid_dataset, test_dataset)
    # Number of params
    model_total_params, model_total_trainable_params = get_model_params(model)
    # Tensorboard logs
    tensorboard_writer = get_tensorboard_writer(model_name)

    model_info: ModelInfo = {
        "model": model,
        "model_params": model_params,
        "model_name": model_name,
        "optimizer": optimizer,
        "criterion": criterion,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "device": device,
        "tensorboard_writer": tensorboard_writer,
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader,
        "model_total_params": model_total_params,
        "model_total_trainable_params": model_total_trainable_params
    }

    write_model_info_to_tensorboard(model_info,
                                    train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    test_dataset=test_dataset)

    return model_info
