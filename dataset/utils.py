import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Optional

from dataset.diagnosis import INFO


def get_transform(args):
    transform = transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            transforms.PILToTensor(),
        ]
    )
    return transform

def get_prototype(args):
    text_classes = list(INFO[args.dataset.lower()]["label"].values())
    return text_classes

@dataclass
class DiagnosisDataCollator:
    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [instance['pixel_values'] for instance in instances]    # List of image tensors
        labels = [instance['label'] for instance in instances]    # List of label arrays

        pixel_values = torch.stack(images)                        # Shape: (batch_size, C, H, W)

        if isinstance(labels[0], torch.Tensor) and labels[0].ndim == 0:
            labels = [int(label) for label in labels]
        else:
            labels = [int(label[0]) if isinstance(label, (list, tuple, np.ndarray, torch.Tensor)) else int(label) for label in labels]
        labels = torch.tensor(labels, dtype=torch.long)           # Shape: (batch_size,)

        batch = {
            'pixel_values': pixel_values,
            'labels': labels
        }

        return batch