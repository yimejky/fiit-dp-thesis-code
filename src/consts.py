import sys

from src.helpers.get_bounding_box import get_dividable_bounding_box

MAX_PADDING_SLICES = 160
DATASET_MAX_BOUNDING_BOX = [56, 177, 156]
DATASET_PADDING_VALUE = -2**10 # 1024 # hounsfield unit min
IN_COLAB = 'google.colab' in sys.modules

# [56 177 156] is bounding box size without spinal cord in dataset, so we get bounding box which can be divided
# by pooling/unpooling layers and in the end still persist size
DESIRE_BOUNDING_BOX_SIZE = get_dividable_bounding_box(DATASET_MAX_BOUNDING_BOX, pooling_layers=3, offset=12)

