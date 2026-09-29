"""Patho-R1-7B adapter for the Qwen2.5-VL benchmark interface.

Official inference recipe: https://github.com/Wenchuan-Zhang/Patho-R1
"""

import os
import re
import warnings

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from model.chat import ChatMetaModel

OFFICIAL_REASONING_PROMPT = (
    "You are a pathology expert, your task is to answer question step by step. "
    "Use the following format:<think> Your step-by-step reasoning </think>"
    "<answer> Your final answer </answer>"
)


def _pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    return to_pil_image(image.detach().cpu()).convert("RGB")


def final_answer(response):
    """Remove official reasoning tags for direct VQA scoring only."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.I | re.S)
    if match:
        return match.group(1).strip()
    return re.sub(r"<think>.*?</think>", "", response, flags=re.I | re.S).strip()


class PathoR1(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)
        self.name = "Patho-R1-7B"
        self.model_type = "medical"
        self.prefers_cpu_image_inputs = True

    def load_from_pretrained(self, model_path, **kwargs):
        self.set_device(kwargs.get("device", getattr(self.args, "device", "cuda")))
        model_base = getattr(self.args, "model_base", None)
        adapter = os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "adapter_config.json"))
        load_path = model_base if adapter else model_path
        if adapter and not model_base:
            raise ValueError("Patho-R1 LoRA checkpoint requires --model_base pointing to the authorized Patho-R1-7B weights.")
        for attention in ("flash_attention_2", "sdpa", "eager"):
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    load_path, torch_dtype="auto", device_map=str(self.device), attn_implementation=attention
                )
                break
            except ImportError:
                if attention == "eager":
                    raise
                warnings.warn(f"Patho-R1: {attention} unavailable; trying fallback")
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(load_path)
        self.tokenizer = self.processor.tokenizer

    def load_for_training(self, model_path):
        if not getattr(self.args, "patho_r1_training_authorized", False):
            raise PermissionError(
                "Patho-R1 checkpoint terms prohibit derivative training without prior written approval. "
                "Obtain that approval before passing --patho_r1_training_authorized True."
            )
        if getattr(self.args, "peft", None) != "lora":
            raise ValueError("Patho-R1 framework training currently supports --peft lora only.")
        if self.args.bits != 16:
            raise ValueError("Patho-R1 framework training currently requires --bits 16.")
        modules = self.args.tune_modules.upper()
        if not modules or set(modules) - set("VML"):
            raise ValueError("--tune_modules must contain V, M, or L.")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.args.bf16 else torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model.config.use_cache = False
        self.model.requires_grad_(False)
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        target_names = []
        for name, layer in self.model.named_modules():
            if not isinstance(layer, torch.nn.Linear):
                continue
            is_merger = "visual.merger" in name
            if (("V" in modules and "visual." in name and not is_merger)
                    or ("M" in modules and is_merger)
                    or ("L" in modules and "model.layers." in name)):
                target_names.append(name)
        if target_names:
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    bias=self.args.lora_bias,
                    target_modules=target_names,
                    task_type="CAUSAL_LM",
                ),
            )
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("No trainable Patho-R1 parameters selected.")

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        images = image if isinstance(image, list) else [image]
        messages = [
            {"role": "system", "content": (
                "You are a pathology expert" if getattr(self.args, "usage", None) in ("mdagent", "ucagent")
                else OFFICIAL_REASONING_PROMPT
            )},
            {"role": "user", "content": [
                *({"type": "image", "image": _pil(item)} for item in images),
                {"type": "text", "text": qs},
            ]}
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generation = {"max_new_tokens": 2048, "do_sample": temperature is not None and temperature > 0}
        if generation["do_sample"]:
            generation["temperature"] = temperature
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        answer_ids = generated[0][inputs.input_ids.shape[1]:]
        response = self.processor.decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        return final_answer(response)

    def save(self, output_folder, trainer=None):
        self.model.save_pretrained(output_folder)
        self.processor.save_pretrained(output_folder)
