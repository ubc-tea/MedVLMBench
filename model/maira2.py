import os
import warnings
import torch
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

from model.chat import ChatMetaModel


class MAIRA2(ChatMetaModel):
    """Wrapper for microsoft/maira-2, a grounded chest X-ray report generator.

    Unlike the other generative wrappers in this repo, MAIRA-2 is not a general
    chat/VQA model: it exposes a structured `format_and_preprocess_reporting_input`
    API keyed on radiology-report fields (indication/technique/comparison/prior
    report) rather than a free-form chat template. To fit MedVLMBench's generic
    (image, question) interface, the incoming `qs` is passed through as the
    `indication` field and the other report fields are left blank; for the
    model's intended use (structured findings generation) prefer calling
    `generate_report` directly with real clinical fields.

    Gated on Hugging Face (Microsoft Research License Agreement, non-commercial).
    """

    def __init__(self, args):
        super().__init__(args)

        self.name = "MAIRA-2"
        self.model_type = "medical"

    def load_from_pretrained(self, model_path, **kwargs):
        device = kwargs.get("device", getattr(self.args, "device", "cuda"))
        self.set_device(device)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def infer_vision_language(self, image, qs, image_size=None, temperature=None):
        if not type(image) == list:
            image = [image]

        if len(image) > 1:
            warnings.warn(
                f"MAIRA-2 received {len(image)} images; using the first as the current frontal "
                "view and the second (if present) as the current lateral view.",
                stacklevel=2,
            )

        frontal = to_pil_image(image[0]).convert("RGB")
        lateral = to_pil_image(image[1]).convert("RGB") if len(image) > 1 else None

        return self.generate_report(frontal, lateral=lateral, indication=qs)

    def generate_report(
        self,
        frontal,
        lateral=None,
        prior_frontal=None,
        prior_report=None,
        indication="",
        technique="",
        comparison="",
        get_grounding=False,
    ):
        processed_inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=frontal,
            current_lateral=lateral,
            prior_frontal=prior_frontal,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_report=prior_report,
            return_tensors="pt",
            get_grounding=get_grounding,
        ).to(self.model.device)

        output_decoding = self.model.generate(
            **processed_inputs,
            max_new_tokens=450 if get_grounding else 300,
            use_cache=True,
        )

        prompt_length = processed_inputs["input_ids"].shape[-1]
        decoded_text = self.processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)

        return self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

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
                warnings.warn(f"[MAIRA-2] Image path not found: {candidate_path}")
                continue
            try:
                with Image.open(candidate_path) as img:
                    loaded_images.append(img.convert("RGB"))
            except Exception as exc:
                warnings.warn(f"[MAIRA-2] Failed to open image {candidate_path}: {exc}")

        return loaded_images
