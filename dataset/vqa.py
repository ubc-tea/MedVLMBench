import os
import json
import pandas as pd
import numpy as np
from PIL import Image

from datasets import load_dataset, concatenate_datasets
from dataset.base import BaseDataset


def _resolve_fairvlmed10k_roots(image_path):
    image_root = os.path.abspath(image_path)
    metadata_candidates = [
        image_root,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "FairVLMed10k"),
    ]

    metadata_root = None
    for candidate in metadata_candidates:
        if os.path.exists(os.path.join(candidate, "vqa_test.csv")):
            metadata_root = candidate
            break

    if metadata_root is None:
        metadata_root = image_root

    return metadata_root, image_root


class VQADataset(BaseDataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split)

        if transform is not None:
            print(f"Loading image processor: {transform}")

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
            "question_type": "open" if is_open else "yes/no",
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
            "question_type": "open" if is_open else "yes/no",
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
            "question_type": "open" if is_open else "yes/no",
            "prompt_template": prompt_template,
            "image_size": image_size,
            "image_path": image_path,
        }


class HarvardFairVLMed10kVQA(VQADataset):
    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        self.name = "Harvard-FairVLMed10k"
        self.modality = "SLO Fundus"

        self.metadata_path, self.image_path = _resolve_fairvlmed10k_roots(data_args.image_path)
        self.ds = pd.read_csv(os.path.join(self.metadata_path, f"vqa_{split}.csv")).dropna()

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

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "query": qs,
            "label": answer,
            "is_open": is_open,
            "question_type": "open" if is_open else "yes/no",
            "prompt_template": prompt_template,
            "image_size": image_size,
            "image_path": image_path,
        }


class MedXpertQA(VQADataset):
    _SPLIT_MAP = {
        "train": ["cus_train_fold0"],
        "test": ["cus_test_fold0"],
        "all": ["test"],  # official split for testing (N=2000)
    }

    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        if split not in self._SPLIT_MAP:
            raise ValueError(f"Unsupported split '{split}' for MedXpertQA")

        self.name = "MedXpertQA-MM"
        self.modality = "medical"
        self.data_dir = data_args.image_path
        self.image_root = os.path.join(self.data_dir, "images")
        self.annotation_root = os.path.join(self.data_dir, "MM")

        self.samples = []
        for subset in self._SPLIT_MAP[split]:
            annotation_path = os.path.join(self.annotation_root, f"{subset}.jsonl")
            if not os.path.exists(annotation_path):
                raise FileNotFoundError(f"MedXpertQA annotation file not found: {annotation_path}")

            with open(annotation_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    record["_subset"] = subset
                    self.samples.append(record)

    def __len__(self):
        return len(self.samples)

    def _load_and_merge_images(self, image_files):
        images = []
        paths = []
        for image_file in image_files:
            path = os.path.join(self.image_root, image_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"MedXpertQA image not found: {path}")
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
            paths.append(path)

        if not images:
            raise RuntimeError("No images found for the given image files.")

        return images, paths, [x.size for x in images]

        # combine multiple images into one
        # if len(images) == 1:
        #     return images[0], paths, images[0].size

        # target_height = max(img.height for img in images)
        # resized = []
        # total_width = 0
        # for img in images:
        #     if img.height != target_height:
        #         new_width = int(img.width * target_height / img.height)
        #         img = img.resize((new_width, target_height), Image.BICUBIC)
        #     resized.append(img)
        #     total_width += img.width

        # canvas = Image.new("RGB", (total_width, target_height), color=(255, 255, 255))
        # x_offset = 0
        # for img in resized:
        #     canvas.paste(img, (x_offset, 0))
        #     x_offset += img.width

        # return canvas, paths, canvas.size

    def __getitem__(self, index):
        sample = self.samples[index]

        question = sample["question"].strip()
        answer = sample["label"].strip()
        options = sample["options"]
        image_files = sample.get("images", [])

        images, image_paths, image_sizes = self._load_and_merge_images(image_files)
        relative_image_paths = [os.path.relpath(path, start=self.data_dir) for path in image_paths]

        prompt_template = "{}\nAnswer with the single letter corresponding to the best choice."  # we can consider putting it in argument.

        if self.transform is not None:
            images = [self.transform(image) for image in images]

        return {
            "image": images,
            "query": question,
            "label": answer,  # letter only
            "is_open": False,  # MedXpertQA is multiple-choice; treat as closed-form
            "question_type": "multi-choice",
            "prompt_template": prompt_template,  # '{}\nAnswer with the single letter corresponding to the best choice.'
            "image_size": image_sizes,
            "image_path": ";".join(image_paths),
            "image_paths": image_paths,
            "options": options,  # {"A": "...", "B": "...", ...}
        }


class OmniMedVQA(VQADataset):
    _INDEX_FILES = {
        "train": "omnimedvqa_train_index.jsonl",
        "validation": "omnimedvqa_train_index.jsonl",  # reuse train split when val is requested
        "test": "omnimedvqa_test_index.jsonl",
    }

    def __init__(self, data_args, split, transform=None):
        super().__init__(data_args, split, transform)

        if split not in ["train", "validation", "test", "all"]:
            raise ValueError(f"Unsupported split '{split}' for OmniMedVQA")

        self.name = "OmniMedVQA"
        self.modality = "medical"

        self.data_dir = data_args.image_path
        self.qa_root = os.path.join(self.data_dir, "QA_information")

        if not os.path.isdir(self.qa_root):
            fallback_root = self._locate_nested_root(self.data_dir)
            if fallback_root is None:
                raise FileNotFoundError(f"OmniMedVQA QA information directory not found: {self.qa_root}")
            self.data_dir = fallback_root
            self.qa_root = os.path.join(self.data_dir, "QA_information")

        records = self._load_records()

        if split != "all":
            index_set = self._load_split_index(split)
            records = [rec for rec in records if rec["question_id"] in index_set]

            missing = len(index_set) - len(records)
            if missing > 0:
                print(
                    f"[OmniMedVQA] Warning: {missing} entries from {split} index not found with accessible images. "
                    "Ensure dataset paths are correct."
                )

        self.samples = records

    def _locate_nested_root(self, base_dir):
        try:
            for entry in os.listdir(base_dir):
                candidate = os.path.join(base_dir, entry)
                if not os.path.isdir(candidate):
                    continue
                qa_dir = os.path.join(candidate, "QA_information")
                if os.path.isdir(qa_dir):
                    return candidate
        except FileNotFoundError:
            return None
        return None

    def _load_records(self):
        samples = []
        missing_images = 0
        for access_type in ["Open-access", "Restricted-access"]:
            dir_path = os.path.join(self.qa_root, access_type)
            if not os.path.isdir(dir_path):
                continue
            for filename in sorted(os.listdir(dir_path)):
                if not filename.endswith(".json"):
                    continue
                annotation_path = os.path.join(dir_path, filename)
                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f)

                for record in annotations:
                    rel_path = record.get("image_path", "").strip()
                    abs_path = os.path.normpath(os.path.join(self.data_dir, rel_path))
                    if not os.path.exists(abs_path):
                        missing_images += 1
                        continue
                    record["_abs_image_path"] = abs_path
                    record["_access_type"] = access_type
                    samples.append(record)

        if not samples:
            raise RuntimeError(
                "No OmniMedVQA samples found with accessible images. "
                "Please ensure the dataset is correctly downloaded and the image paths are valid."
            )

        if missing_images > 0:
            print(f"[OmniMedVQA] Skipped {missing_images} annotations due to missing image files.")

        return samples

    def _load_split_index(self, split):
        """
        Load pre-generated train/test indices so splits are fixed across machines.
        """
        index_filename = self._INDEX_FILES.get(split)
        if index_filename is None:
            raise ValueError(f"Unsupported split '{split}' for OmniMedVQA")

        # OmniMedVQA splits are pinned to resample_v4 to keep evaluation consistent.
        candidate_dirs = []
        for candidate_dir in [
            self.data_dir,
            os.path.dirname(self.data_dir),
            os.path.join(self.data_dir, "resample_v4"),
            os.path.join(os.path.dirname(self.data_dir), "resample_v4"),
        ]:
            normalized = os.path.normpath(candidate_dir)
            if normalized not in candidate_dirs:
                candidate_dirs.append(normalized)

        searched = []
        for candidate_dir in candidate_dirs:
            candidate = os.path.join(candidate_dir, index_filename)
            searched.append(candidate)
            if not os.path.isfile(candidate):
                continue

            index_set = set()
            with open(candidate, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = entry.get("question_id")
                    if qid:
                        index_set.add(qid)
            if not index_set:
                raise RuntimeError(f"Index file {candidate} is empty.")
            return index_set

        raise FileNotFoundError(
            f"Index file for split '{split}' not found. Searched: {', '.join(searched)}."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = sample["_abs_image_path"]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image_size = image.size

        question = sample["question"].strip()
        answer_text = sample.get("gt_answer", "").strip()

        option_keys = sorted(k for k in sample.keys() if k.startswith("option_"))
        options = []
        answer_letter = None
        normalized_answer = answer_text.lower()
        for key in option_keys:
            letter = key.split("_")[-1]
            text = str(sample[key]).strip()
            options.append((letter, text))
            if normalized_answer and normalized_answer == text.lower():
                answer_letter = letter.upper()

        prompt_template = "{}"
        is_open = True

        if options:
            option_lines = "\n".join(f"({letter}) {text}" for letter, text in options)
            prompt_template = (
                "{}\nOptions:\n" + option_lines + "\nAnswer with the single letter corresponding to the best choice."
            )
            if answer_letter is None:
                # fall back to textual answer if it does not match provided options
                answer_letter = answer_text.strip()
            is_open = len(answer_letter) != 1 or not answer_letter.isalpha()
        else:
            answer_letter = answer_text.strip()

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "query": question,
            "label": answer_letter,  # TODO answer_letter is a bad choice and need to be updated.
            "is_open": is_open,  # Not important here
            "question_type": "multi-choice",
            "prompt_template": prompt_template,  # '{}\nOptions:\n(A) Biopsy\n(B) CT scan\n(C) Colonoscopy\n(D) Fundus imaging\nAnswer with the single letter corresponding to the best choice.'
            "image_size": image_size,
            "image_path": image_path,
            "options": {letter: text for letter, text in options},
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
