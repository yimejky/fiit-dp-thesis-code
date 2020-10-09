import SimpleITK as sitk
import numpy as np

import src.helpers.oars_labels_consts as OARS_LABELS

from scipy import ndimage
from torch.utils.data import Dataset
from pathlib import Path
from functools import reduce


class HaNOarsDataset(Dataset):
    """ source https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """

    def __init__(self, root_dir, size, shrink_factor=1):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir_path = Path(root_dir)
        self.size = size
        self.shrink_factor = shrink_factor

        self.data_list = []
        self.label_list = []
        self.load_images()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_list[idx], self.label_list[idx]

    def load_images(self):
        for i in range(self.size):
            data_filepath = Path.joinpath(self.root_dir_path, f'./{i + 1}/data.nii.gz')
            label_filepath = Path.joinpath(self.root_dir_path, f'./{i + 1}/label.nii.gz')

            data = sitk.ReadImage(str(data_filepath))
            label = sitk.ReadImage(str(label_filepath))

            self.data_list.append(data)
            self.label_list.append(label)

        return self

    def filter_labels(self, wanted_labels):
        for i in range(self.size):
            label = self.label_list[i]

            # acum = 0
            # for label_val in wanted_labels:
            #     acum = acum | (label == label_val)
            # self.label_list[i] = acum

            self.label_list[i] = reduce(lambda a, b: a | (label == b), wanted_labels, 0)

        return self

    def shrink(self):
        for i in range(self.size):
            data = self.data_list[i]
            label = self.label_list[i]

            shrink_filter = sitk.ShrinkImageFilter()
            shrink_size = (self.shrink_factor, self.shrink_factor, 1)

            self.data_list[i] = shrink_filter.Execute(data, shrink_size)
            self.label_list[i] = shrink_filter.Execute(label, shrink_size)

        return self

    def data_normalize(self):
        for i in range(self.size):
            data = self.data_list[i]

            USE_NUMPY = True
            if USE_NUMPY:
                data_np = sitk.GetArrayFromImage(data)

                data_np = (data_np - data_np.mean()) / data_np.std()
                # data_np = (data_np - data_np.min()) / (data_np.max() - data_np.min())

                self.data_list[i] = sitk.GetImageFromArray(data_np)
            else:  # simple itk
                norm_filter = sitk.NormalizeImageFilter()
                self.data_list[i] = norm_filter.Execute(data)

        return self

    def dilatate_labels(self, repeat=1):
        for i in range(self.size):
            label = self.label_list[i]

            USE_NUMPY = True
            if USE_NUMPY:
                label_np = sitk.GetArrayFromImage(label)
                for j in range(repeat):
                    label_np = ndimage.binary_dilation(label_np).astype(label_np.dtype)
                self.label_list[i] = sitk.GetImageFromArray(label_np)
            else:  # simple itk
                label_tmp = label
                for j in range(repeat):
                    label_tmp = sitk.BinaryDilateImageFilter().Execute(label)
                self.label_list[i] = label_tmp

        return self

    def to_sitk(self):
        for i in range(self.size):
            data_np = self.data_list[i]
            label_np = self.label_list[i]

            data = sitk.GetImageFromArray(data_np[0])
            label = sitk.GetImageFromArray(label_np)

            self.data_list[i] = data
            self.label_list[i] = label

        return self

    def to_numpy(self):
        for i in range(self.size):
            data = self.data_list[i]
            label = self.label_list[i]

            data_np = sitk.GetArrayFromImage(data)
            label_np = sitk.GetArrayFromImage(label)

            # adding channel dimension, because conv3d => channel, slices, height, width
            self.data_list[i] = np.expand_dims(data_np, axis=0)
            self.label_list[i] = label_np

        return self


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = HaNOarsDataset(f'./data/{"HaN_OAR"}_shrink{16}x_padded160', 5)
    dataset.data_normalize()
    dataset.filter_labels([OARS_LABELS.EYE_L, OARS_LABELS.EYE_R, OARS_LABELS.LENS_L, OARS_LABELS.LENS_R])
    dataset.to_numpy()

    data = dataset.data_list[0]
    label = dataset.label_list[0]
    slice_n = 99

    # print(data.shape, data.min()data_normalize, data.max())
    # plt.imshow(data[0, slice_n], cmap="gray")
    # plt.show()
    #
    print(label.shape, label.min(), label.max(), label[label > 0].shape[0])
    plt.imshow(label[slice_n], cmap="gray")
    plt.show()

    #
    # tmp = ndimage.binary_dilation(label).astype(label.dtype)
    # print(tmp.shape, tmp.min(), tmp.max())
    # plt.imshow(tmp[slice_n], cmap="gray")
    # plt.show()
