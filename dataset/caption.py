import os
import json
import pandas as pd
import numpy as np
from easydict import EasyDict as edict

from PIL import Image

from datasets import load_dataset, concatenate_datasets
from dataset.base import BaseDataset


class CaptionDataset(BaseDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split)

        self.transform = transform
        self.prompt_template = "Can you provide a medical report for this image? {}"


class HarvardFairVLMed10kCaption(CaptionDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "Harvard-FairVLMed10k"
        self.modality = "SLO Fundus"

        self.image_path = data_args.image_path
        self.ds = pd.read_csv(os.path.join("./data/FairVLMed10k", f"caption_{split}.csv"))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds.iloc[index]

        image_path = os.path.join(self.image_path, item["image_path"])
        image = Image.fromarray(np.load(image_path)["slo_fundus"]).convert("RGB")
        image_size = image.size

        caption = item["gpt_summary"]
        prompt_template = self.prompt_template

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": caption,
            "prompt_template": prompt_template,
            "image_size": image_size,
            "image_path": image_path,
        }


class MIMIC_CXRCaption(CaptionDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "MIMIC-CXR"
        self.modality = "Chest Xray"

        self.image_path = os.path.join(data_args.image_path, "images")

        with open(os.path.join(data_args.image_path, f"annotation.json"), "r") as f:
            annotations = json.load(f)

        self.data = [item for item in annotations[self.split]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        image_paths = item["image_path"]
        image_path = os.path.join(self.image_path, image_paths[0])
        image = Image.open(image_path).convert("RGB")
        image_size = image.size

        if self.transform:
            image = self.transform(image)

        caption = item["report"]

        return {
            "image": image,
            "label": caption,
            "prompt_template": self.prompt_template,
            "image_size": image_size,
            "image_path": image_path,
        }


if __name__ == "__main__":
    dataset = MIMIC_CXRCaption(edict(image_path="/fast/rjin02/DataSets/mimic_cxr"), split="train", transform=None)
    sample = dataset[0]
    print("Image Path:", sample["image_path"])
    print("Label:", sample["label"])
