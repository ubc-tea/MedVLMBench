import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel


class _AttentionOutputLoRA(nn.Module):
    """Update the weight read directly by torch MultiheadAttention."""

    def __init__(self, weight, rank=8, alpha=32):
        super().__init__()
        self.A = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.B = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, weight):
        return weight + (self.B @ self.A) * self.scale


def _upgrade_legacy_attention_lora(module, state_dict, prefix, *args):
    """Read checkpoints with PEFT out_proj keys from older runs."""
    suffixes = {
        ".attn.out_proj.base_layer.weight": ".attn.out_proj.parametrizations.weight.original",
        ".attn.out_proj.base_layer.bias": ".attn.out_proj.bias",
        ".attn.out_proj.lora_A.default.weight": ".attn.out_proj.parametrizations.weight.0.A",
        ".attn.out_proj.lora_B.default.weight": ".attn.out_proj.parametrizations.weight.0.B",
    }
    for key in list(state_dict):
        if not key.startswith(prefix):
            continue
        for old, new in suffixes.items():
            if key.endswith(old):
                state_dict[key[: -len(old)] + new] = state_dict.pop(key)
                break


class OpenCLIPMixin:
    """Shared plumbing for CLIP-style models that expose the `open_clip` interface
    (`encode_image` / `encode_text` / `visual` / `logit_scale`).

    Subclasses implement `build_model()` returning ``(model, preprocess)`` and
    `tokenize(texts)` returning an integer token tensor.
    """

    embed_dim = 512

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.usage in {"clip-img-lora", "img-lora-lp"}:
            attention_modules = [
                module for module in self.model.visual.modules()
                if isinstance(module, nn.MultiheadAttention)
            ]
            for attention in attention_modules:
                parametrize.register_parametrization(
                    attention.out_proj,
                    "weight",
                    _AttentionOutputLoRA(attention.out_proj.weight),
                )
            if attention_modules:
                self.model.register_load_state_dict_pre_hook(_upgrade_legacy_attention_lora)
                self.args.logger.info(
                    "Enabled %d attention output LoRA parametrizations",
                    len(attention_modules),
                )

    def find_target_linear_names(self, model, *args, **kwargs):
        verbose = kwargs.pop("verbose", True)
        names = super().find_target_linear_names(model, *args, verbose=False, **kwargs)
        skipped = {
            f"{name}.out_proj" for name, module in model.named_modules()
            if isinstance(module, nn.MultiheadAttention)
        }
        names = [name for name in names if name not in skipped]
        if verbose:
            self.args.logger.info("Found %d PEFT LoRA modules: %s", len(names), names)
        return names

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
