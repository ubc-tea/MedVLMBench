import torch
import torch.nn.functional as F
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel


class OpenCLIPMixin:
    """Shared plumbing for CLIP-style models that expose the `open_clip` interface
    (`encode_image` / `encode_text` / `visual` / `logit_scale`).

    Subclasses implement `build_model()` returning ``(model, preprocess)`` and
    `tokenize(texts)` returning an integer token tensor.
    """

    embed_dim = 512

    def build_model(self):
        raise NotImplementedError

    def tokenize(self, texts):
        raise NotImplementedError

    def setup_encoders(self):
        self.model.vision_model = self.model.visual
        self.text_embed_dim = self.embed_dim
        self.vision_embed_dim = self.embed_dim

    def _image_features(self, images):
        return self.model.encode_image(images)

    def _text_features(self, tokens):
        return self.model.encode_text(tokens)


class OpenCLIPForDiagnosis(OpenCLIPMixin, CLIPBase):
    """Zero-shot (`clip-zs`) and image-LoRA (`clip-img-lora`) wrapper."""

    def __init__(self, text, num_classes, args=None, *kargs, **kwargs):
        model, preprocess = self.build_model()
        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)

        self.image_processor = ImageProcessorCallable(preprocess)
        self.image_processor_evaluation = self.image_processor
        self.initialize_prototypes()

    def initialize_prototypes(self):
        if self.prototype is None:
            self.prototype = self.tokenize(self.prototype_text).to(self.args.device)

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        return self._text_features(self.tokenize(text).to(device))

    def encode_image(self, images):
        return self._image_features(images)

    def forward(self, pixel_values):
        image_features = F.normalize(self._image_features(pixel_values), dim=-1)
        with torch.no_grad():
            text_features = F.normalize(self._text_features(self.prototype.to(pixel_values.device)), dim=-1)
        return self.model.logit_scale.exp() * image_features @ text_features.t()


class OpenCLIPLPForDiagnosis(OpenCLIPMixin, CLIPImgLPModel):
    """Linear-probe (`lp`) and LoRA + linear-probe (`img-lora-lp`) wrapper."""

    def __init__(self, args, text, num_classes) -> None:
        model, preprocess = self.build_model()
        super().__init__(text=text, num_classes=num_classes, model=model, args=args)

        self.image_processor = ImageProcessorCallable(preprocess)
        self.image_processor_evaluation = self.image_processor

    def encode_image(self, images):
        return self._image_features(images)
