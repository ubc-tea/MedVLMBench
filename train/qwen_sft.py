# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example commands (LoRA rank 128) using preprocessed JSONs in script/:
- Qwen2.5 + MedXpertQA-MM:
  accelerate launch --main_process_port 49000 train/qwen_sft.py --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct --dataset_name medxpertqa_mm_train_qwen --output_dir ./log/sft_qwen25_medxpertqa_lora128 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 1 --learning_rate 2e-5 --max_seq_length 2048 --logging_steps 10 --save_steps 500 --bf16 --lora_enable True --lora_rank 128 --lora_alpha 256
- Qwen2.5 + OmniMedVQA:
  accelerate launch --main_process_port 49001 train/qwen_sft.py --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct --dataset_name omnimedvqa_train_qwen --output_dir ./log/sft_qwen25_omnimedvqa_lora128 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 1 --learning_rate 2e-5 --max_seq_length 2048 --logging_steps 10 --save_steps 500 --bf16 --lora_enable True --lora_rank 128 --lora_alpha 256
- Lingshu + MedXpertQA-MM:
  accelerate launch --main_process_port 49002 train/qwen_sft.py --model_name_or_path lingshu-medical-mllm/Lingshu-7B --dataset_name medxpertqa_mm_train_qwen --output_dir ./log/sft_lingshu_medxpertqa_lora128 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 1 --learning_rate 2e-5 --max_seq_length 2048 --logging_steps 10 --save_steps 500 --bf16 --lora_enable True --lora_rank 128 --lora_alpha 256
- Lingshu + OmniMedVQA:
  accelerate launch --main_process_port 49003 train/qwen_sft.py --model_name_or_path lingshu-medical-mllm/Lingshu-7B --dataset_name omnimedvqa_train_qwen --output_dir ./log/sft_lingshu_omnimedvqa_lora128_v2 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 1 --learning_rate 2e-5 --max_seq_length 2048 --logging_steps 10 --save_steps 500 --bf16 --lora_enable True --lora_rank 128 --lora_alpha 256
  accelerate launch --main_process_port 49003 train/qwen_sft.py --model_name_or_path lingshu-medical-mllm/Lingshu-7B --dataset_name omnimedvqa_train_qwen --output_dir ./log/sft_lingshu_omnimedvqa_lora128_v3 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 1 --learning_rate 2e-5 --max_seq_length 2048 --logging_steps 10 --save_steps 500 --bf16 --lora_enable True --lora_rank 128 --lora_alpha 256
  accelerate launch --main_process_port 49003 train/qwen_sft.py --model_name_or_path lingshu-medical-mllm/Lingshu-7B --dataset_name omnimedvqa_train_qwen --output_dir ./log/sft_lingshu_omnimedvqa_lora128_v4 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --num_train_epochs 1 --learning_rate 2e-5 --max_seq_length 2048 --logging_steps 10 --save_steps 500 --bf16 --lora_enable True --lora_rank 128 --lora_alpha 256

"""
import ast
import json
import torch
import random
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor, default_data_collator
from qwen_vl_utils import process_vision_info

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field

from transformers import AutoModelForCausalLM, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLConfig, Qwen2VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Qwen2_5_VLProcessor
AutoModelForCausalLM.register(config_class=Qwen2_5_VLConfig, model_class=Qwen2_5_VLForConditionalGeneration)
AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)

from torch.utils.data import Dataset

from PIL import Image

@dataclass
class ExtendedModelConfig(ModelConfig):
    lora_enable: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    lora_namespan_exclude: str = "[]"  # A stringified list; will be parsed in the code.
    freeze_llm: bool = False


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

CHAT_TEMPLATE = {
    "chat_template": "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
}


def report_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num = param.numel()
            trainable_params += num
            print(f" - {name} | shape: {tuple(param.shape)} | params: {num:,}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({(trainable_params / total_params) * 100:.2f}%)")

# oracle answer 
def make_conversation(example):
    """
    This method will add images according to the number in the original data
    """ 
    content = []
    for img in example["images"]:
        img_dict = {
                        "type": "image",
                        "image": img,
                        "resized_height": 336,
                        "resized_width": 336,
                    }
        content.append(img_dict)
    
    content.append(
        {
            "type": "text",
            "text": example["messages"][0]["content"],
        }
    )

    return [
                {
                    "role": "user",
                    "content": content,
                },
                {
                    "role": "assistant",
                    "content": 
                    [
                        {
                            "type": "text",
                            "text": example["messages"][1]["content"]
                        }
                    ],
                },
            ]

def seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(script_args, training_args, model_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    def load_conversation_dataset(name: str):
        """
        Resolve dataset JSON and return the conversation list expected by make_conversation.
        Supports:
          - Absolute or relative paths ending with .json
          - Known med/retina/derma prefixes (existing behavior)
          - Preprocessed MedXpertQA/OmniMedVQA JSONs placed in ../script/
        """
        if name.endswith(".json"):
            dataset_path = name
        elif name in ["medxpertqa_mm_train_qwen", "omnimedvqa_train_qwen"]:
            dataset_path = f"./script/{name}.json"
        else:
            dataset_path = f"./data/{name}.json"

        with open(dataset_path, "r") as f:
            sat_dataset = json.load(f)
        return [make_conversation(sample) for sample in sat_dataset]

    if script_args.dataset_name is not None:
        dataset = load_conversation_dataset(script_args.dataset_name)
    else:
        raise NotImplementedError()

    if len(dataset) == 0:
        raise ValueError(
            f"Resolved dataset '{script_args.dataset_name}' is empty. "
            "Check the source JSON and dataset preparation step."
        )

    dataset = CustomDataset(dataset)

    ################
    # Define processor
    ################
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(
                example,
                tokenize=False,
                add_generation_prompt=True
            )
            for example in examples
        ]

        image_inputs_list = []
        video_inputs_list = []

        for example in examples:
            # process_vision_info is now expected to return a tuple: (image_inputs, video_inputs)
            img_inputs, vid_inputs = process_vision_info(example)
            image_inputs_list.append(img_inputs)
            video_inputs_list.append(vid_inputs)

        batch = processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655] # <|vision_start|> <|vision_end|> <|vision_pad|>
        elif isinstance(processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels
            
        batch["labels"] = labels  # Add labels to the batch

        return batch

    ################
    # Training
    ################
    model_name_lower = model_args.model_name_or_path.lower()
    if "qwen2.5" in model_name_lower or "lingshu" in model_name_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
        )



    if getattr(model_args, "lora_enable", False):
        from peft import LoraConfig, get_peft_model
        # Ensure lora_namespan_exclude is a list.
        if model_args.lora_namespan_exclude is not None:
            model_args.lora_namespan_exclude = ast.literal_eval(model_args.lora_namespan_exclude)
        else:
            model_args.lora_namespan_exclude = []
        # If vision LoRA is not enabled, exclude visual modules.
        if not getattr(model_args, "vision_lora", False):
            model_args.lora_namespan_exclude += ["visual"]

        def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
            linear_cls = torch.nn.Linear
            embedding_cls = torch.nn.Embedding
            lora_module_names = []
            for name, module in model.named_modules():
                if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
                    continue
                if isinstance(module, (linear_cls, embedding_cls)):
                    lora_module_names.append(name)
            if num_lora_modules > 0:
                lora_module_names = lora_module_names[-num_lora_modules:]
            if verbose:
                print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
            return lora_module_names

        target_modules = find_target_linear_names(
            model,
            num_lora_modules=getattr(model_args, "num_lora_modules", -1),
            lora_namespan_exclude=model_args.lora_namespan_exclude,
        )

        peft_config = LoraConfig(
            r=getattr(model_args, "lora_rank", 8),
            lora_alpha=getattr(model_args, "lora_alpha", 32),
            target_modules=target_modules,
            lora_dropout=getattr(model_args, "lora_dropout", 0.1),
            bias=getattr(model_args, "lora_bias", "none"),
        )
        print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # model.visual.requires_grad_ = True
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, padding_side='right')
    # processor.chat_template = CHAT_TEMPLATE["chat_template"]

    training_args.ddp_find_unused_parameters=False
    training_args.model_init_kwargs = None
    training_args.dataset_text_field = ""
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    trainer_peft_config = None if getattr(training_args, "lora_enable", False) else get_peft_config(model_args)

    report_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=trainer_peft_config,
        tokenizer=processor.tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()

    # Save and push to hub
    processor.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ExtendedModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # print(training_args)
    seed(training_args.seed)
    main(script_args, training_args, model_args)
