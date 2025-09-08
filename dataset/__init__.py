import os
import random
import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from collections import Counter

from dataset.utils import get_transform
from dataset.vqa import SLAKE, PathVQA, VQARAD, HarvardFairVLMed10kVQA
from dataset.caption import HarvardFairVLMed10kCaption, MIMIC_CXRCaption
from dataset.diagnosis import PneumoniaMNIST, BreastMNIST, DermaMNIST, Camelyon17, HAM10000Dataset, DrishtiDataset, ChestXrayDataset, GF3300Dataset, CXPDataset, PAPILADataset, FairVLMed10kDataset

datasets = {
    "SLAKE-vqa": SLAKE,
    "PathVQA-vqa": PathVQA,
    "VQA-RAD-vqa": VQARAD,
    "Harvard-FairVLMed10k-vqa": HarvardFairVLMed10kVQA,
    "MIMIC_CXR-caption": MIMIC_CXRCaption,
    "PneumoniaMNIST-diagnosis": PneumoniaMNIST,
    "BreastMNIST-diagnosis": BreastMNIST,
    "DermaMNIST-diagnosis": DermaMNIST,
    "Camelyon17-diagnosis": Camelyon17,
    "HAM10000-diagnosis": HAM10000Dataset,
    "Drishti": DrishtiDataset,
    "ChestXray-diagnosis": ChestXrayDataset,
    "GF3300-diagnosis": GF3300Dataset,
    "HarvardFairVLMed10k-caption": HarvardFairVLMed10kCaption,
    "CheXpert-diagnosis": CXPDataset,
    "PAPILA-diagnosis": PAPILADataset,
    "HarvardFairVLMed10k-diagnosis": FairVLMed10kDataset,
}


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
    
    if image_processor_callable is not None:
        transform = image_processor_callable
    else:
        transform = get_transform(args)

    dataset = dataset_name(data_args=edict(image_path=args.image_path, size=224), split=split, transform=transform)

    try:
        args.logger.info("Loaded dataset: " + dataset.name)
        args.logger.info(f"Dataset size: {len(dataset)}")
    except:
        print("Logger is not set.")

    if args.task == "diagnosis":
        report_label_distribution(dataset, args)

    return dataset


def report_label_distribution(dataset, args):
    label_counts = Counter()
    for i in range(len(dataset)):
        label = dataset[i]["label"].item()
        label_counts[label] += 1

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
