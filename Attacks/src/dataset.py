import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np
import pandas as pd
from torchvision import transforms


class TaskDataset(Dataset):
    def __init__(self, path, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform
        self.path = path

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.2980, 0.2962, 0.2987], std=[0.2886, 0.2875, 0.2889]
                )  # Normalizes each channel
            ]
        )
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


data: MembershipDataset = torch.load("./pub.pt", weights_only=False)
