import concurrent.futures

import SimpleITK as sitk
import numpy as np
from torchio.transforms import (
    ZNormalization,
    RandomAffine,
    Compose,
)
import matplotlib.pyplot as plt

import src.dataset.oars_labels_consts as OARS_LABELS
from src.consts import CPU_COUNT

from scipy import ndimage
from torch.utils.data import Dataset
from pathlib import Path
from functools import reduce

from src.dataset.transform_input import transform_input


class HaNOarsDataset(Dataset):
    """ source https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """

    def __init__(self,
                 root_dir,
                 size,
                 shrink_factor=1,
                 load_images=True,
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir_path = Path(root_dir)
        self.size = size
        self.shrink_factor = shrink_factor
        self.transform = transform

        self.is_numpy = False
        self.output_label = None
        self.data_list = []
        self.label_list = []
        if load_images:
            self.load_images()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item_data = self.data_list[idx]
        item_label = self.label_list[idx]

        # label filtering
        if self.output_label is not None:
            if type(self.output_label) is list:
                item_label = reduce(lambda a, b: a | (item_label == b), self.output_label, False) * 1
            else:
                item_label = (item_label == self.output_label) * 1
            item_label = item_label.astype(np.int8)

        # torchio data augmentation and transforms
        if self.transform is not None:
            item_data, item_label = transform_input(item_data, item_label, self.transform)

        return item_data, item_label

    def set_output_label(self, output_label=None):
        self.output_label = output_label

    def copy(self, copy_lists=True):
        # creating obj
        copy_dataset = HaNOarsDataset(
            str(self.root_dir_path),
            self.size,
            self.shrink_factor,
            load_images=False)

        # coping inner attributes
        copy_dataset.is_numpy = self.is_numpy
        copy_dataset.output_label = self.output_label

        # coping data and labels
        copy_dataset.data_list = [None] * self.size
        copy_dataset.label_list = [None] * self.size
        if copy_lists:
            for i in range(self.size):
                print(f'copying index {i}')
                copy_dataset.data_list[i] = self.data_list[i].copy()
                copy_dataset.label_list[i] = self.label_list[i].copy()

        return copy_dataset

    def get_data_size(self):
        bytes_size = 0
        for i in range(self.size):
            bytes_size += self.data_list[i].nbytes + self.label_list[i].nbytes
        return bytes_size

    def show_data_type(self):
        if self.is_numpy:
            print('data type:', self.data_list[0].dtype, self.label_list[0].dtype)
        else:
            print('data type:', self.data_list[0].GetPixelIDTypeAsString(), self.label_list[0].GetPixelIDTypeAsString())

    def load_images(self):
        for i in range(self.size):
            data_filepath = Path.joinpath(self.root_dir_path, f'./{i + 1}/data.nii.gz')
            label_filepath = Path.joinpath(self.root_dir_path, f'./{i + 1}/label.nii.gz')

            data = sitk.ReadImage(str(data_filepath))
            label = sitk.ReadImage(str(label_filepath))

            self.data_list.append(data)
            self.label_list.append(label)

        return self

    def filter_labels(self, wanted_labels, unify_labels):
        print('filtering labels')

        def filter_label(input_label, index):
            USE_NUMPY = True
            if USE_NUMPY:
                label = sitk.GetArrayFromImage(input_label).astype(np.int8)
                label_idx = reduce(lambda a, b: a | (label == b), wanted_labels, False)
                new_label = np.zeros(label.shape)
                if unify_labels:
                    new_label[label_idx] = 1
                else:
                    new_label[label_idx] = label[label_idx]
                self.label_list[index] = sitk.GetImageFromArray(new_label.astype(np.int8))
            else:
                label = input_label
                self.label_list[index] = reduce(lambda a, b: a | (label == b), wanted_labels, 0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
            [executor.submit(filter_label, self.label_list[i], i) for i in range(self.size)]
        print('filtering labels done')

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

    def data_normalize__deprecated_(self):
        # __DEPRECATED__
        print('normalizing dataset')

        def normalize(data, index):
            USE_NUMPY = True
            if USE_NUMPY:
                data_np = sitk.GetArrayFromImage(data)
                data_np = (data_np - data_np.mean()) / data_np.std()
                # data_np = (data_np - data_np.min()) / (data_np.max() - data_np.min())
                self.data_list[index] = sitk.GetImageFromArray(data_np)
            else:  # simple itk
                norm_filter = sitk.NormalizeImageFilter()
                self.data_list[index] = norm_filter.Execute(data)

        with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
            [executor.submit(normalize, self.data_list[i], i) for i in range(self.size)]
        print("normalizing done")

        return self

    def dilatate_labels(self, repeat=1):
        print(f'dilatating {repeat}x dataset')
        for i in range(self.size):
            label = self.label_list[i]

            USE_NUMPY = True
            if USE_NUMPY:
                label_np = sitk.GetArrayFromImage(label).astype(np.int8)
                for j in range(repeat):
                    label_np = ndimage.binary_dilation(label_np).astype(np.int8)
                self.label_list[i] = sitk.GetImageFromArray(label_np)
            else:  # simple itk
                label_tmp = label
                for j in range(repeat):
                    label_tmp = sitk.BinaryDilateImageFilter().Execute(label)
                self.label_list[i] = label_tmp

        return self

    def to_sitk(self):
        self.is_numpy = False
        print('parsing dataset to simple itk')
        for i in range(self.size):
            data_np = self.data_list[i]
            label_np = self.label_list[i]

            data = sitk.GetImageFromArray(data_np[0])
            label = sitk.GetImageFromArray(label_np.astype(np.int8))

            self.data_list[i] = data
            self.label_list[i] = label

        return self

    def to_numpy(self):
        self.is_numpy = True
        print('parsing dataset to numpy')

        def parse_data(data, index):
            data_np = sitk.GetArrayFromImage(data)
            # adding channel dimension, because conv3d takes: channel, slices, height, width
            self.data_list[index] = np.expand_dims(data_np, axis=0)

        def parse_label(label, index):
            label_np = sitk.GetArrayFromImage(label)
            self.label_list[index] = label_np.astype(np.int8)

        with concurrent.futures.ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
            [executor.submit(parse_data, self.data_list[i], i) for i in range(self.size)]
            [executor.submit(parse_label, self.label_list[i], i) for i in range(self.size)]
        print("numpy parsing done")

        return self


if __name__ == "__main__":
    slice_n = 99

    spatial = RandomAffine(
        scales=1,
        degrees=3,
        isotropic=False,
        default_pad_value='otsu',
        image_interpolation='bspline')
    transform = Compose([
        spatial,
        ZNormalization()
    ])

    dataset = HaNOarsDataset(f'./data/{"HaN_OAR"}_shrink{2}x_padded160', 10, transform=transform)
    dataset.filter_labels([OARS_LABELS.EYE_L, OARS_LABELS.EYE_R, OARS_LABELS.LENS_L, OARS_LABELS.LENS_R], False)
    dataset.to_numpy()

    tmp_data, tmp_label = dataset[0]

    # tmp_label2 = dataset.label_list[0]
    # unique, counts = np.unique(tmp_label[tmp_label > 0], return_counts=True)
    # print(np.asarray((unique, counts)).T)
    # unique, counts = np.unique(tmp_label2[tmp_label2 > 0], return_counts=True)
    # print(np.asarray((unique, counts)).T)

    print(tmp_data.shape, tmp_data.min(), tmp_data.max())
    print(tmp_label.shape, tmp_label.min(), tmp_label.max())
    print(np.unique(tmp_label))

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(tmp_data[0, slice_n], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(tmp_label[slice_n], cmap="gray")
    plt.show()

    # tmp_img = (tmp_label == 1) + (tmp_label == 2) * 2
    # tmp_img_np = sitk.GetArrayFromImage(tmp_img)
    # print(np.unique(tmp_img))
    # plt.imshow(tmp_img_np[100])
    # plt.show()
    #
    # tmp_data = dataset.data_list[0]
    # tmp_label = dataset.label_list[0]
    #
    # print(tmp_data.shape, tmp_data.min()data_normalize, tmp_data.max())
    # plt.imshow(tmp_data[0, slice_n], cmap="gray")
    # plt.show()
    #
    # print(tmp_label.shape, tmp_label.min(), tmp_label.max(), tmp_label[tmp_label > 0].shape[0])
    # plt.imshow(tmp_label[slice_n], cmap="gray")
    # plt.show()
