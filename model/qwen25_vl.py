import os
import shutil
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from model.chat import ChatMetaModel


class Qwen25_VL(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "Qwen2.5-VL"
        self.model_type = "general"

    def load_from_pretrained(self, model_path, **kwargs):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def infer_vision_language(self, image, qs, image_size=None):
        image = to_pil_image(image)

        # prepare messages
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": qs}]}]
        print(messages)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(output_text)

        return output_text[0].strip()
