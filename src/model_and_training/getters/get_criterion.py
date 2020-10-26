from src.losses.dice_loss import DiceLoss


def get_criterion():
    criterion = DiceLoss()
    # criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    return criterion
