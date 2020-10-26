from src.model_and_training.unet_architecture_v2 import UNetV2


def get_model(device, model_params):
    # model = UNet(**model_params).to(device)
    model = UNetV2(**model_params).to(device)

    return model
