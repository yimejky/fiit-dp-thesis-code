import logging

from torch import nn
from src.helpers.calc_dsc import calc_dsc


class DiceLoss(nn.Module):
    """ https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py """
    """ https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/ """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_true):
        logging.debug(f'DiceLoss1 y_true {y_true.shape} y_pred, {y_pred.shape}')
        assert y_pred.size() == y_true.size()
        dsc = calc_dsc(y_true, y_pred, self.smooth)

        logging.debug(f'DiceLoss2 smooth {self.smooth}, dsc {dsc}, 1-dsc {1 - dsc}')

        return 1. - dsc
