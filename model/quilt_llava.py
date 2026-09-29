"""Quilt-LLaVA-v1.5-7B uses the released LLaVA-1.5-7B architecture."""

from model.llava import LLaVA


class QuiltLLaVA(LLaVA):
    def __init__(self, args):
        super().__init__(args)
        self.conv_mode = "llava_v1"
        self.name = "Quilt-LLaVA-v1.5"
        self.model_type = "medical"

    def load_from_pretrained(self, model_path, **kwargs):
        super().load_from_pretrained(model_path, **kwargs)
        # The released model_vqa.py evaluates the checkpoint with the CLIP
        # image processor directly, rather than applying training-time pad.
        self.image_processor_callable = self._preprocess_official_vqa

    def _preprocess_official_vqa(self, image):
        return self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    def _preprocess_inference_image(self, image):
        return self._preprocess_official_vqa(image)
