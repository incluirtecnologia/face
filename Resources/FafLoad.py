import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class FafLoad(Dataset):
    def __init__(self, csv_file, root, transforms, label, rotulos = None):
        rotulos = '/rotulos.txt' if not rotulos else rotulos
        self.root = root
        self.transforms = transforms
        self.label = label
        # load all image files, sorting them to
        # ensure that they are aligned
        listImgs = pd.read_csv(csv_file)
        self.imgs = list(listImgs.iloc[:, 0])
        targets = list(np.loadtxt(self.root + '/rotulos.txt'))
        targets = [label if x==1 else 0 for x in targets]
        self.targets = targets

    def __getitem__(self, idx):
        # load images and masks
        img = Image.open(self.imgs[idx]).convert("RGB")

        boxes = [0, 0, img.width, img.height]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = self.targets[idx]

        image_id = torch.tensor([idx])
        area = img.width * img.height
        # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)
