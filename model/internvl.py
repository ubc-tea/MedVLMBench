import os
import shutil
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModelForImageTextToText

from model.chat import ChatMetaModel


class InternVL3(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "InternVL3"
        self.model_type = "general"

    def load_from_pretrained(self, model_path, **kwargs):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
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
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        generate_ids = self.model.generate(**inputs, max_new_tokens=200)
        decoded = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(decoded)

        return decoded.strip()
