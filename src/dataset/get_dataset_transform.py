from torchio import RandomAffine, Compose

from src.dataset.get_norm_transform import get_norm_transform


def get_dataset_transform():
    spatial = RandomAffine(
        scales=1,
        degrees=3,
        isotropic=False,
        default_pad_value='otsu',
        image_interpolation='bspline')
    transform = Compose([
        spatial,
        get_norm_transform()
    ])

    return transform

