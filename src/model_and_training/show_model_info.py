from torchsummary import summary

from src.consts import IN_COLAB, MAX_PADDING_SLICES


def show_model_info(model_info):
    if IN_COLAB:
        summary(model_info["model"], (1, MAX_PADDING_SLICES, 128, 128), batch_size=1)

    print(f'Model number of params: {model_info["model_total_params"]}, trainable {model_info["model_total_trainable_params"]}')
