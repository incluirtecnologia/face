import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io, transform
import time, os
import numpy as np
from skimage import io
from PIL import Image


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.targets = list(np.loadtxt(self.root_dir + '/rotulos.txt'))
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = self.targets
        self.img_labels = self.targets


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()
      # if torch.is_tensor(idx):
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
      image = io.imread(self.landmarks_frame.iloc[idx, 0])
      if self.transform:
        image = self.transform(image)
        # image = image.transpose((2, 0, 1))
        # image = torch.from_numpy(image)
      labels = self.targets[idx]
      labels = np.array([labels])
      sample = {'image': image, 'labels': labels}
      return sample

