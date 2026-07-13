import os
import tempfile
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from model.chat import ChatMetaModel


class CheXagent2(ChatMetaModel):
    """Wrapper for StanfordAIMI/CheXagent-2-3b, a chest X-ray specialist VLM.

    Custom `CheXagentForCausalLM` architecture (Phi-based LM + XraySigLIP vision
    encoder). Its tokenizer exposes `from_list_format`, which expects image *paths*
    rather than in-memory PIL images, so incoming tensors are written to a temporary
    directory before formatting the prompt.
    """

    def __init__(self, args):
        super().__init__(args)

        self.name = "CheXagent-2"
        self.model_type = "medical"
        self._tmp_dir = None

    def load_from_pretrained(self, model_path, **kwargs):
        device = kwargs.get("device", getattr(self.args, "device", "cuda"))
        self.set_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
            .to(torch.bfloat16)
            .eval()
        )

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        if not type(image) == list:
            image = [image]

        pil_images = [to_pil_image(x).convert("RGB") for x in image]

        self._tmp_dir = tempfile.mkdtemp(prefix="chexagent2_")
        image_paths = []
        for idx, img in enumerate(pil_images):
            path = os.path.join(self._tmp_dir, f"image_{idx}.png")
            img.save(path)
            image_paths.append(path)

        try:
            query = self.tokenizer.from_list_format(
                [*[{"image": path} for path in image_paths], {"text": qs}]
            )
            conv = [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            gen_kwargs = dict(do_sample=False, num_beams=1, temperature=1.0, top_p=1.0, use_cache=True)
            if temperature is not None and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)

            output = self.model.generate(input_ids, max_new_tokens=512, **gen_kwargs)[0]
            response = self.tokenizer.decode(output[input_ids.size(1) : -1])
        finally:
            for path in image_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            try:
                os.rmdir(self._tmp_dir)
            except OSError:
                pass

        return response.strip()

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
                warnings.warn(f"[CheXagent-2] Image path not found: {candidate_path}")
                continue
            try:
                with Image.open(candidate_path) as img:
                    loaded_images.append(img.convert("RGB"))
            except Exception as exc:
                warnings.warn(f"[CheXagent-2] Failed to open image {candidate_path}: {exc}")

        return loaded_images
