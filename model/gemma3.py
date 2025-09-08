import os
import shutil
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from model.chat import ChatMetaModel


class Gemma3(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "Gemma3"
        self.model_type = "general"

    def load_from_pretrained(self, model_path, **kwargs):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def infer_vision_language(self, image, qs, image_size=None):
        image = to_pil_image(image)

        # prepare messages
        messages = [
            # {
            #     "role": "system",
            #     "content": [{"type": "text", "text": "You are a helpful assistant."}],
            # },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": qs},
                    {"type": "image", "image": image},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        # print(decoded)

        return decoded.strip()
