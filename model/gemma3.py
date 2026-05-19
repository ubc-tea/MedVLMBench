import os
import shutil
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image

from model.chat import ChatMetaModel


class Gemma3(ChatMetaModel):
    def __init__(self, args):
        super().__init__(args)

        self.name = "Gemma3"
        self.model_type = "general"
        self.prefers_cpu_image_inputs = True

    def load_from_pretrained(self, model_path, **kwargs):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        if not type(image) == list:
            image = [image]
        image_contents = [{"type": "image", "image": to_pil_image(x)} for x in image]

        # prepare messages
        messages = [
            # {
            #     "role": "system",
            #     "content": [{"type": "text", "text": "You are a helpful assistant."}],
            # },
            {
                "role": "user",
                "content": [
                    *image_contents,
                    {"type": "text", "text": qs},
                ],
            },
        ]

        inputs = self._prepare_inputs(messages).to(self.model.device, dtype=torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            if temperature is None:
                generation = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
            else:
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                )

            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        # print(decoded)

        return decoded.strip()

    def _prepare_inputs(self, messages):
        chat_template = getattr(self.processor, "chat_template", None)
        if chat_template:
            return self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )

        prompt, images = self._build_default_prompt(messages)
        processor_kwargs = {
            "text": [prompt],
            "padding": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if images:
            processor_kwargs["images"] = [images]

        return self.processor(**processor_kwargs)

    def _build_default_prompt(self, messages, add_generation_prompt=True):
        """
        Fallback prompt builder for processor checkpoints that do not ship a chat template
        (e.g. Gemma-3 PT variants). It formats messages using the simple "role\\ncontent"
        layout recommended by Google and inserts image placeholders so the processor can
        inject vision embeddings.
        """
        boi_token = getattr(self.processor, "boi_token", "<start_of_image>")
        role_map = {"assistant": "model", "user": "user"}

        prompt_segments = []
        images = []

        for message in messages:
            role = role_map.get(message.get("role", "user"), message.get("role", "user"))
            prompt_segments.append(f"{role}\n")
            for content in message.get("content", []):
                content_type = content.get("type")
                if content_type == "text":
                    text = content.get("text", "")
                    if text:
                        prompt_segments.append(text.rstrip())
                        prompt_segments.append("\n")
                elif content_type == "image":
                    prompt_segments.append(f"{boi_token}\n")
                    image_value = content.get("image")
                    if image_value is not None:
                        images.append(image_value)
            prompt_segments.append("\n")

        if add_generation_prompt:
            prompt_segments.append("model\n")

        prompt = "".join(prompt_segments)
        return prompt, images

    def _load_context_images(self):
        context = getattr(self, "_inference_context", None) or {}
        image_paths = context.get("image_paths")
        self._inference_context = {}

        if not image_paths:
            return []

        if isinstance(image_paths, str):
            image_paths = [p for p in image_paths.split(";") if p]

        loaded_images = []
        for image_path in image_paths:
            candidate_path = image_path
            if not os.path.isabs(candidate_path):
                base_dir = getattr(self.args, "image_path", "") or ""
                candidate_path = os.path.join(base_dir, image_path)
            if not os.path.exists(candidate_path):
                warnings.warn(f"[Gemma3] Image path not found: {candidate_path}")
                continue
            try:
                with Image.open(candidate_path) as img:
                    loaded_images.append(img.convert("RGB"))
            except Exception as exc:
                warnings.warn(f"[Gemma3] Failed to open image {candidate_path}: {exc}")

        return loaded_images
