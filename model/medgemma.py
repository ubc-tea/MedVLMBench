import os
import shutil
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from model.chat import ChatMetaModel
from peft import LoraConfig


class MedGemma(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "MedGemma"
        self.model_type = "medical"

    def load_from_pretrained(self, model_path, **kwargs):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def load_for_training(self, model_path):
        # Check if GPU supports bfloat16
        if torch.cuda.get_device_capability()[0] < 8:
            raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")

        model_kwargs = dict(
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )

        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "right"

        modules_to_save = []
        exclude_modules = ""
        if "M" in self.args.tune_modules:
            modules_to_save.extend(
                [
                    "multi_modal_projector"
                    # "lm_head",
                    # "embed_tokens",
                ]
            )

        # print(self.model)

        if "V" not in self.args.tune_modules:
            exclude_modules = r".*\.vision_tower\..*"
        if "L" not in self.args.tune_modules:
            exclude_modules = r".*\.language_model\.layers\..*"

        self.peft_config = LoraConfig(
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            r=self.args.lora_r,
            bias=self.args.lora_bias,
            target_modules="all-linear",
            exclude_modules=exclude_modules,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,
        )

    def infer_vision_language(self, image, qs, image_size=None):
        image = to_pil_image(image)

        # prepare messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert on understanding medical images."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": qs},
                    {"type": "image", "image": image},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)

        print(decoded)

        return decoded.strip()

    def save(self, output_folder, trainer=None):
        trainer.save_model()
