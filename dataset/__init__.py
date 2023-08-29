# dataset/__init__.py


import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from typing import Tuple


class EEGWindowDataset(Dataset):
    def __init__(self, train_set_file: str, labels_file: str,
                 window_size: int = 256, transform=None, target_transform=None):
        self.train_set = pd.read_csv(train_set_file).rename(columns={"Time:512Hz": "time"})
        self.labels = pd.read_csv(labels_file).set_index("ID").angle

        self.window_size = window_size

        sets = list(self.train_set.groupby(["Subject", "Session"]))

        self.IDs = []

        for (subj, _), x in sets:
            self.IDs.extend(x.ID.tolist()[:-self.window_size])

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.IDs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, float]:
        ID = self.IDs[idx]

        x = torch.tensor(
            self.train_set.iloc[ID:ID + self.window_size].drop(columns=["ID", "Subject", "Session", "time"]).values
        )
        if self.transform:
            x = self.transform(x)

        label = self.labels[ID]
        if self.target_transform:
            label = self.target_transform(label)

        return x, label


def train_val_dataset(dataset, val_split: float = 0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return datasets


def load_eeg_windows(train_set_file: str, labels_file: str,
                     window_size: int = 256, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    dataset = EEGWindowDataset(train_set_file, labels_file, window_size=window_size)
    datasets = train_val_dataset(dataset, val_split=val_split)

    train_loader = DataLoader(datasets['train'], 32, shuffle=True, num_workers=4)
    val_loader = DataLoader(datasets['val'], 32, shuffle=True, num_workers=4)
    return train_loader, val_loader
