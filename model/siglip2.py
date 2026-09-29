"""SigLIP2 diagnosis wrappers using the published dual encoder."""

from transformers import AutoProcessor, Siglip2Model

from model.clip_base import CLIPBase, CLIPImgLPModel, ImageProcessorCallable


_BACKBONE = "google/siglip2-base-patch16-224"


class _SigLIP2Mixin:
    def setup_encoders(self):
        self.vision_embed_dim = self.model.config.vision_config.hidden_size
        self.text_embed_dim = self.model.config.text_config.hidden_size

    def encode_image(self, images):
        return self.model.get_image_features(pixel_values=images)


class SigLIP2ForDiagnosis(_SigLIP2Mixin, CLIPBase):
    def __init__(self, args, text, num_classes):
        source = getattr(args, "vision_backbone", None) or _BACKBONE
        super().__init__(
            args=args, text=text, num_classes=num_classes,
            model=Siglip2Model.from_pretrained(source),
        )
        processor = AutoProcessor.from_pretrained(source)
        self.tokenizer = processor.tokenizer
        self.image_processor = ImageProcessorCallable(processor.image_processor)
        self.image_processor_evaluation = self.image_processor
        self.initialize_prototypes()

    def forward(self, pixel_values):
        tokens = self.prototype.to(pixel_values.device)
        return self.model(
            pixel_values=pixel_values, input_ids=tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
        ).logits_per_image


class SigLIP2LPForDiagnosis(_SigLIP2Mixin, CLIPImgLPModel):
    def __init__(self, args, text, num_classes):
        source = getattr(args, "vision_backbone", None) or _BACKBONE
        super().__init__(
            args=args, text=text, num_classes=num_classes,
            model=Siglip2Model.from_pretrained(source),
        )
        processor = AutoProcessor.from_pretrained(source)
        self.image_processor = ImageProcessorCallable(processor.image_processor)
        self.image_processor_evaluation = self.image_processor
