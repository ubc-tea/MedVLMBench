from typing import Dict, Optional, Sequence, List

import copy
import transformers
import tokenizers
from dataclasses import dataclass, field
from easydict import EasyDict as edict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import model.release.llava.conversation as conversation_lib
from model.release.llava.mm_utils import tokenizer_image_token

from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def preprocess_multimodal(args, query, answer, constants):
    sources = [
        {"from": "human", "value": query},
        {
            "from": "gpt",
            "value": answer,
        },
    ]

    for sentence in sources:
        if constants.DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = sentence["value"].replace(constants.DEFAULT_IMAGE_TOKEN, "").strip()
            sentence["value"] = constants.DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in args.version:
                sentence["value"] = sentence["value"].replace(
                    constants.DEFAULT_IMAGE_TOKEN, "<Image>" + constants.DEFAULT_IMAGE_TOKEN + "</Image>"
                )
        replace_token = constants.DEFAULT_IMAGE_TOKEN
        if args.mm_use_im_start_end:
            replace_token = constants.DEFAULT_IM_START_TOKEN + replace_token + constants.DEFAULT_IM_END_TOKEN
        sentence["value"] = sentence["value"].replace(constants.DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources, conv, tokenizer: transformers.PreTrainedTokenizer, model_constants, has_image: bool = False
) -> Dict:
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    if roles[sources[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        sources = sources[1:]

    conv.messages = []
    for j, sentence in enumerate(sources):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])
    conversations = [conv.get_prompt()]

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = model_constants.IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = model_constants.IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = model_constants.IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = model_constants.IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources, conv, tokenizer: transformers.PreTrainedTokenizer, model_constants, has_image: bool = False
) -> Dict:
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    if roles[sources[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        sources = sources[1:]

    conv.messages = []
    for j, sentence in enumerate(sources):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])

    conversations = [conv.get_prompt()]

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = model_constants.IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = model_constants.IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = model_constants.IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = model_constants.IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    args, sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, model_constants, has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    conv = conversation_lib.conv_templates[args.version]
    if conv.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, conv.copy(), tokenizer, model_constants, has_image=has_image)
    if conv.version.startswith("v1"):
        return preprocess_v1(sources, conv.copy(), tokenizer, model_constants, has_image=has_image)
    if conv.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conv.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LLaVADataset(Dataset):
    def __init__(self, args, base_dataset, tokenizer, image_processor, model_constants):
        super().__init__()

        self.args = args
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_constants = model_constants

    def __len__(self):
        return len(self.base_dataset)

    @property
    def lengths(self):
        length_list = []
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            img_tokens = 128 if "image" in sample.keys() else 0
            cur_len = len(sample["label"].split()) + len(sample["prompt_template"].split())
            if "query" in sample.keys():
                cur_len += len(sample["query"].split())
            length_list.append(cur_len + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            cur_len = len(sample["label"].split()) + len(sample["prompt_template"].split())
            if "query" in sample.keys():
                cur_len += len(sample["query"].split())
            cur_len = cur_len if "image" in sample.keys() else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.base_dataset[i]

        image = sources["image"]
        query = sources["query"] if "query" in sources.keys() else ""
        answer = sources["label"]
        prompt_template = sources["prompt_template"]

        image = transforms.functional.to_pil_image(image, mode="RGB")

        if self.args.image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = self.image_preprocess(image, return_tensors="pt")["pixel_values"][0]

        query = query.replace(self.model_constants.DEFAULT_IMAGE_TOKEN, "").strip()
        query = prompt_template.format(query) + "/n" + self.model_constants.DEFAULT_IMAGE_TOKEN
        sources = preprocess_multimodal(self.args, query, answer, self.model_constants)

        data_dict = preprocess(self.args, sources, self.tokenizer, self.model_constants, has_image=image is not None)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.base_dataset[i]:
            data_dict["image"] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model_constants: edict

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.model_constants.IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


def make_supervised_data_module(
    args, dataset, tokenizer: transformers.PreTrainedTokenizer, image_processor, model_constants
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LLaVADataset(args, dataset, tokenizer, image_processor, model_constants)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, model_constants=model_constants)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
