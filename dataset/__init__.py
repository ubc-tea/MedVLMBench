import os
import random
import numpy as np
import pandas as pd
import torch
import json
from easydict import EasyDict as edict
from collections import Counter

from dataset.utils import get_transform
from dataset.vqa import SLAKE, PathVQA, VQARAD, HarvardFairVLMed10kVQA, MedXpertQA, OmniMedVQA
from dataset.caption import HarvardFairVLMed10kCaption, MIMIC_CXRCaption
from dataset.diagnosis import (
    PneumoniaMNIST,
    BreastMNIST,
    DermaMNIST,
    Camelyon17,
    HAM10000Dataset,
    DrishtiDataset,
    ChestXrayDataset,
    GF3300Dataset,
    CXPDataset,
    PAPILADataset,
    FairVLMed10kDataset,
)

datasets = {
    "SLAKE-vqa": SLAKE,
    "PathVQA-vqa": PathVQA,
    "VQA-RAD-vqa": VQARAD,
    "Harvard-FairVLMed10k-vqa": HarvardFairVLMed10kVQA,
    "MedXpertQA-vqa": MedXpertQA,
    "OmniMedVQA-vqa": OmniMedVQA,
    "MIMIC_CXR-caption": MIMIC_CXRCaption,
    "PneumoniaMNIST-diagnosis": PneumoniaMNIST,
    "BreastMNIST-diagnosis": BreastMNIST,
    "DermaMNIST-diagnosis": DermaMNIST,
    "Camelyon17-diagnosis": Camelyon17,
    "HAM10000-diagnosis": HAM10000Dataset,
    "Drishti-diagnosis": DrishtiDataset,
    "ChestXray-diagnosis": ChestXrayDataset,
    "GF3300-diagnosis": GF3300Dataset,
    "HarvardFairVLMed10k-caption": HarvardFairVLMed10kCaption,
    "CheXpert-diagnosis": CXPDataset,
    "PAPILA-diagnosis": PAPILADataset,
    "HarvardFairVLMed10k-diagnosis": FairVLMed10kDataset,
}


class FractionalDataset(torch.utils.data.Dataset):
    """A deterministic subset that preserves benchmark dataset metadata."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
        for attr in ("name", "modality", "split"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __getattr__(self, name):
        if name in {"dataset", "indices"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


def _apply_train_fraction(dataset, args, split):
    fraction = float(getattr(args, "train_fraction", 1.0))
    if split != "train" or fraction == 1.0:
        return dataset
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must satisfy 0 < train_fraction <= 1")

    generator = torch.Generator().manual_seed(int(getattr(args, "fraction_seed", 42)))
    subset_size = max(1, int(round(len(dataset) * fraction)))
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].sort().values.tolist()
    subset = FractionalDataset(dataset, indices)

    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "dataset": getattr(dataset, "name", getattr(args, "dataset", None)),
            "split": split,
            "source_size": len(dataset),
            "selected_size": len(subset),
            "fraction_requested": fraction,
            "fraction_realized": len(subset) / len(dataset),
            "fraction_seed": int(getattr(args, "fraction_seed", 42)),
            "indices": indices,
        }
        with open(os.path.join(output_dir, "train_subset_manifest.json"), "w") as fp:
            json.dump(payload, fp, indent=2)
    return subset


def get_dataset(args, image_processor_callable=None, split=None):

    g = torch.Generator()
    g.manual_seed(args.seed)

    def seed_worker(worker_id):
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset_name = datasets[f"{args.dataset}-{args.task}"]

    assert args.split in ["train", "validation", "test", "all"]

    if split is None:
        assert args.split in ["train", "validation", "test", "all"]
        split = args.split

    assert image_processor_callable is not None or args.task != "diagnosis"

    llava_train_models = {"LLaVA-1.5", "LLaVA-Med", "Quilt-LLaVA"}

    # LLaVA training performs its own image padding and preprocessing in the
    # trainer dataset wrapper. Passing a transform here causes double-processing
    # and type mismatches for PIL/tensor/BatchFeature inputs.
    if getattr(args, "model", None) in llava_train_models and args.task in {"vqa", "caption"} and split == "train":
        transform = None
    elif image_processor_callable is not None:
        transform = image_processor_callable
    else:
        transform = get_transform(args)

    dataset = dataset_name(data_args=edict(image_path=args.image_path, size=224), split=split, transform=transform)
    dataset = _apply_train_fraction(dataset, args, split)

    try:
        args.logger.info("Loaded dataset: " + dataset.name)
        args.logger.info(f"Dataset size: {len(dataset)}")
    except:
        print("Logger is not set.")

    if args.task == "diagnosis":
        report_label_distribution(dataset, args)

    return dataset


def report_label_distribution(dataset, args):
    source = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", None)
    labels = getattr(source, "Y", None)
    if labels is None and getattr(source, "name", None) == "Camelyon17":
        labels = source.data["tumor"].to_numpy()
    if labels is not None:
        selected = labels if indices is None else labels[indices]
        label_counts = Counter(int(label) for label in selected)
    else:
        label_counts = Counter(int(dataset[i]["label"].item()) for i in range(len(dataset)))

    total = sum(label_counts.values())
    distribution = {label: count / total for label, count in label_counts.items()}

    args.logger.info("Label Distribution:")
    for label, freq in distribution.items():
        args.logger.info(f"Label {label}: {freq:.2%} ({label_counts[label]} samples)")

    num_classes = max(label_counts.keys()) + 1
    weights = [0.0] * num_classes
    for lbl, cnt in label_counts.items():
        weights[lbl] = total / (cnt * num_classes)

    dataset.class_weights = torch.tensor(weights, dtype=torch.float)
    args.logger.info(f"Class weights: {dataset.class_weights.tolist()}")
