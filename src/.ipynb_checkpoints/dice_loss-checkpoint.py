from torch import nn
from src.helpers.calc_dsc import calc_dsc

class DiceLoss(nn.Module):
    """ https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py """
    """ https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/ """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        return 1. - calc_dsc(y_true, y_pred, self.smooth)
    