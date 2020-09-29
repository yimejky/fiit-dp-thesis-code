import SimpleITK as sitk
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class HaNOarsDataset(Dataset):
    """ source https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """

    def __init__(self, root_dir, size):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir_path = Path(root_dir)
        self.size = size

        self.data_list = []
        self.label_list = []
        for i in range(size):
            data, label = self.parse_image(i)
            self.data_list.append(data)
            self.label_list.append(label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_list[idx], self.label_list[idx]
        
    def parse_image(self, i):
        i = str(i + 1)
        data_filepath = Path.joinpath(self.root_dir_path, f'./{i}/data.nii.gz')
        label_filepath = Path.joinpath(self.root_dir_path, f'./{i}/label.nii.gz')

        data = sitk.ReadImage(str(data_filepath));
        label = sitk.ReadImage(str(label_filepath));

        data_np = sitk.GetArrayFromImage(data)
        label_np = sitk.GetArrayFromImage(label)

        data_np = (data_np - data_np.mean()) / data_np.std()
        #data_np = (data_np - data_np.min()) / (data_np.max() - data_np.min())

        return np.expand_dims(data_np, axis=0), label_np
    