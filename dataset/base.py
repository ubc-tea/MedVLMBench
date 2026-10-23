import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


FAIRVLMED10K_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "FairVLMed10k")


def resolve_fairvlmed10k_roots(image_path, metadata_file):
    """Return (metadata_root, image_root) for FairVLMed10k.

    `image_path` may be the data root (containing `FairVLMed10k/`) or the FairVLMed10k folder itself.
    Metadata CSVs fall back to the repo's `data/FairVLMed10k` when they are not next to the images.
    """
    image_root = os.path.abspath(image_path)
    if os.path.isdir(os.path.join(image_root, "FairVLMed10k")):
        image_root = os.path.join(image_root, "FairVLMed10k")

    for candidate in [image_root, FAIRVLMED10K_DIR]:
        if os.path.exists(os.path.join(candidate, metadata_file)):
            return candidate, image_root

    return image_root, image_root


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
