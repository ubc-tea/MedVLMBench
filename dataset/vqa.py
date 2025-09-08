import os
import pandas as pd
import numpy as np
from PIL import Image

from datasets import load_dataset, concatenate_datasets
from dataset.base import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split)

        self.transform = transform

        # 0 for open question, 1 for yes/no question
        self.prompt_templates = ["{}", "Answer the following question about the image with yes or no. {}"]


class SLAKE(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "SLAKE"
        self.modality = "medical"

        if split == "all":
            df_train = load_dataset("BoKelvin/SLAKE", split="train")
            df_val = load_dataset("BoKelvin/SLAKE", split="validation")
            df_test = load_dataset("BoKelvin/SLAKE", split="test")

            self.ds = concatenate_datasets([df_train, df_val, df_test])
        else:
            self.ds = load_dataset("BoKelvin/SLAKE", split=split)
            # self.ds = load_dataset("BoKelvin/SLAKE", split=split).select(range(200))  # for debug only

        self.ds = self.ds.filter(lambda x: x["q_lang"].startswith("en"))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_args.image_path, self.ds[index]["img_name"])
        qs = self.ds[index]["question"]
        answer = self.ds[index]["answer"]
        is_open = self.ds[index]["answer_type"] == "OPEN"

        is_binary = answer.lower() in ["yes", "no"]
        prompt_template = self.prompt_templates[int(is_binary)]

        image = Image.open(image_path).convert("RGB")
        image_size = image.size

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "query": qs,
            "label": answer,
            "is_open": is_open,
            "prompt_template": prompt_template,
            "image_size": image_size,
            "image_path": image_path,
        }


class PathVQA(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "PathVQA"
        self.modality = "pathology"

        if split == "all":
            df_train = load_dataset("flaviagiammarino/path-vqa", split="train")
            df_val = load_dataset("flaviagiammarino/path-vqa", split="validation")
            df_test = load_dataset("flaviagiammarino/path-vqa", split="test")

            self.ds = concatenate_datasets([df_train, df_val, df_test])
        else:
            self.ds = load_dataset("flaviagiammarino/path-vqa", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        qs = self.ds[index]["question"]
        answer = self.ds[index]["answer"]

        is_open = answer.lower() not in ["yes", "no"]
        image_path = "NA"

        is_binary = answer.lower() in ["yes", "no"]
        prompt_template = self.prompt_templates[int(is_binary)]

        image = self.ds[index]["image"].convert("RGB")
        image_size = image.size

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "query": qs,
            "label": answer,
            "is_open": is_open,
            "prompt_template": prompt_template,
            "image_size": image_size,
            "image_path": image_path,
        }


class VQARAD(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "VQA-RAD"
        self.modality = "radiology"

        if split == "all":
            df_train = load_dataset("flaviagiammarino/vqa-rad", split="train")
            df_test = load_dataset("flaviagiammarino/vqa-rad", split="test")
            self.ds = concatenate_datasets([df_train, df_test])
        else:
            self.ds = load_dataset("flaviagiammarino/vqa-rad", split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        qs = self.ds[index]["question"]
        answer = self.ds[index]["answer"]
        image = self.ds[index]["image"].convert("RGB")

        is_open = answer.lower() not in ["yes", "no"]
        is_binary = answer.lower() in ["yes", "no"]
        prompt_template = self.prompt_templates[int(is_binary)]

        image_size = image.size if hasattr(image, "size") else (None, None)

        if self.transform is not None:
            image = self.transform(image)

        image_path = "NA"

        return {
            "image": image,
            "query": qs,
            "label": answer,
            "is_open": is_open,
            "prompt_template": prompt_template,
            "image_size": image_size,
            "image_path": image_path,
        }


class HarvardFairVLMed10kVQA(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "Harvard-FairVLMed10k"
        self.modality = "SLO Fundus"

        self.image_path = data_args.image_path
        self.ds = pd.read_csv(os.path.join(self.image_path, f"vqa_{split}.csv")).dropna()
        # self.ds = pd.read_csv(os.path.join("./data/FairVLMed10k", f"vqa_{split}.csv")).dropna()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds.iloc[index]

        image_path = os.path.join(self.image_path, item["image_path"])
        image = Image.fromarray(np.load(image_path)["slo_fundus"]).convert("RGB")
        image_size = image.size

        qs = item["question"]
        answer = item["answer"]

        is_open = item["question type"] == "OPEN"
        is_binary = answer.lower() in ["yes", "no"]

        prompt_template = self.prompt_templates[int(is_binary)]

        image_path = "NA"

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "query": qs,
            "label": answer,
            "is_open": is_open,
            "prompt_template": prompt_template,
            "image_size": image_size,
            "image_path": image_path,
        }


if __name__ == "__main__":
    import torch
    from torchvision.transforms import PILToTensor

    from easydict import EasyDict as edict
    from torch.utils.data import DataLoader

    # dataset = SLAKE(edict(image_path="/mnt/hdd/data/SLAKE/imgs"), split="test", transform=PILToTensor())
    # dataset = SLAKE(edict(image_path="./data/SLAKE/imgs"), split="test", transform=PILToTensor())
    # dataset = PathVQA(edict(image_path="./data/PathVQA/imgs"), split="test", transform=PILToTensor())
    dataset = VQARAD(edict(image_path="./data/VQARAD/imgs"), split="test", transform=PILToTensor())

    image, qs, answer, is_open, image_size, image_path = dataset[0]

    dataloader = DataLoader(dataset, batch_size=2)

    for batch in dataloader:
        break

    print(batch[0])
    print(batch[1])
