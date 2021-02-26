from src.model_and_training.unet_architecture_v2 import UNetV2


def get_model(device, model_params, model_class=UNetV2):
    # model = UNet(**model_params).to(device)
    model = model_class(**model_params).to(device)

    return model
