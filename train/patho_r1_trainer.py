"""Supervised LoRA adaptation of the authorized Patho-R1 checkpoint."""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer
from qwen_vl_utils import process_vision_info

from model.patho_r1 import _pil


class PathoR1Dataset:
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        images = sample["image"] if isinstance(sample["image"], list) else [sample["image"]]
        question = sample["prompt_template"].format(sample["query"])
        messages = [{"role": "user", "content": [
            *({"type": "image", "image": _pil(image)} for image in images),
            {"type": "text", "text": question},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = str(sample["label"])
        full_text = prompt + answer + self.processor.tokenizer.eos_token
        image_inputs, video_inputs = process_vision_info(messages)
        common = {"images": image_inputs, "videos": video_inputs, "return_tensors": "pt"}
        prompt_inputs = self.processor(text=[prompt], **common)
        full_inputs = self.processor(text=[full_text], **common)
        prompt_length = prompt_inputs.input_ids.shape[1]
        ids = full_inputs.input_ids.squeeze(0)
        labels = ids.clone()
        labels[:prompt_length] = -100
        if not torch.any(labels != -100):
            raise ValueError("Patho-R1 training example has no unmasked answer tokens")
        result = {
            "input_ids": ids,
            "attention_mask": full_inputs.attention_mask.squeeze(0),
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in full_inputs:
                result[key] = full_inputs[key]
        return result


class PathoR1Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        batch = {
            "input_ids": pad_sequence([item["input_ids"] for item in features], batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence([item["attention_mask"] for item in features], batch_first=True, padding_value=0),
            "labels": pad_sequence([item["labels"] for item in features], batch_first=True, padding_value=-100),
        }
        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            values = [item[key] for item in features if key in item]
            if values:
                batch[key] = torch.cat(values, dim=0)
        return batch


def make_patho_r1_trainer(args, model_wrapped, dataset):
    args.remove_unused_columns = False
    return Trainer(
        model=model_wrapped.model,
        args=args,
        train_dataset=PathoR1Dataset(dataset, model_wrapped.processor),
        data_collator=PathoR1Collator(model_wrapped.tokenizer.pad_token_id),
        tokenizer=model_wrapped.tokenizer,
    )
