from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_args, split):
        super().__init__()

        self.data_args = data_args
        self.name = ""
        self.modality = ""
        self.split = split

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
