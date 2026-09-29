"""Image-only foundation models for diagnosis linear probing.

The model names and backbone choices follow the FairMedFM model inventory.
These encoders have no text tower, so they are exposed only through ``lp``.
"""

import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image

from model.base import BaseModel
from utils.utils import maybe_zero_3


VISION_FOUNDATION_MODELS = {
    "DINOv2": ("hf", "facebook/dinov2-base"),
    "DINOv3": ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    "RAD-DINO": ("hf", "microsoft/rad-dino"),
    "AIMv2": ("aimv2", "apple/aimv2-large-patch14-native"),
    "UNI2": ("uni2", "hf-hub:MahmoodLab/UNI2-h"),
    "Virchow2": ("virchow2", "hf-hub:paige-ai/Virchow2"),
    "Prov-GigaPath": ("timm", "hf_hub:prov-gigapath/prov-gigapath"),
    "RETFound": ("retfound", ""),
}


class _HFImageProcessor:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)
        return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]


class _TimmImageProcessor:
    def __init__(self, model):
        from timm.data import create_transform, resolve_model_data_config

        self.transform = create_transform(**resolve_model_data_config(model), is_training=False)

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)
        return self.transform(image.convert("RGB"))


def _build_backbone(name, args):
    kind, default = VISION_FOUNDATION_MODELS[name]
    source = getattr(args, "vision_backbone", None) or default
    if kind in {"hf", "aimv2"}:
        from transformers import AutoImageProcessor, AutoModel

        if kind == "aimv2":
            from transformers import Aimv2VisionModel

            model = Aimv2VisionModel.from_pretrained(source)
        else:
            model = AutoModel.from_pretrained(source)
        processor = _HFImageProcessor(AutoImageProcessor.from_pretrained(source))
        feature_dim = model.config.hidden_size
    else:
        import timm

        if kind == "uni2":
            model = timm.create_model(
                source, pretrained=True, img_size=224, patch_size=14, depth=24,
                num_heads=24, init_values=1e-5, embed_dim=1536,
                mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU,
                reg_tokens=8, dynamic_img_size=True,
            )
            feature_dim = 1536
        elif kind == "virchow2":
            model = timm.create_model(
                source, pretrained=True, mlp_layer=timm.layers.SwiGLUPacked,
                act_layer=nn.SiLU,
            )
            feature_dim = 2 * model.num_features
        elif kind == "retfound":
            if not source:
                raise ValueError("RETFound requires --vision_backbone pointing to its local checkpoint")
            state = torch.load(source, map_location="cpu", weights_only=False)
            state = state.get("model", state)
            position = state.get("pos_embed")
            if position is None:
                raise ValueError("RETFound checkpoint has no positional embedding")
            architecture = {
                197: "vit_large_patch16_224",
                257: "vit_large_patch14_dinov2",
            }.get(position.shape[1])
            if architecture is None:
                raise ValueError(f"Unsupported RETFound positional embedding shape: {position.shape}")
            model = timm.create_model(
                architecture, pretrained=False, num_classes=0, global_pool="avg"
            )
            state = {key: value for key, value in state.items() if not key.startswith(("head.", "decoder_")) and key != "mask_token"}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f"RETFound checkpoint does not match ViT-L/16: missing={missing}, unexpected={unexpected}"
                )
            feature_dim = model.num_features
        else:
            model = timm.create_model(source, pretrained=True)
            feature_dim = model.num_features
        processor = _TimmImageProcessor(model)
    return model, processor, feature_dim, kind


class VisionFoundationLPForDiagnosis(BaseModel, nn.Module):
    def __init__(self, args, text, num_classes):
        if args.usage != "lp":
            raise ValueError(f"{args.model} supports diagnosis linear probing (lp) only")
        super().__init__(args=args)
        self.name = args.model
        self.num_classes = num_classes
        self.model, self.image_processor, feature_dim, self.kind = _build_backbone(args.model, args)
        self.image_processor_evaluation = self.image_processor
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.head = nn.Linear(feature_dim, num_classes)

    def encode_image(self, images):
        if self.kind in {"hf", "aimv2"}:
            output = self.model(pixel_values=images)
            if getattr(output, "pooler_output", None) is not None:
                return output.pooler_output
            tokens = output.last_hidden_state
            return tokens.mean(dim=1) if self.kind == "aimv2" else tokens[:, 0]
        output = self.model(images)
        if self.kind == "virchow2":
            return torch.cat([output[:, 0], output[:, 5:].mean(dim=1)], dim=-1)
        return output

    def forward(self, images):
        with torch.no_grad():
            features = self.encode_image(images)
        return self.head(features)

    def load_from_pretrained(self, model_path, device, **kwargs):
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if "head.weight" not in state or "head.bias" not in state:
            raise ValueError("Linear-probe checkpoint must include the classifier head")
        self.load_state_dict(state, strict=True)
        self.to(device)

    def get_parameters_info(self):
        all_size = 0
        tuned_size = 0
        tuned_names = []
        for name, parameter in self.named_parameters():
            size = maybe_zero_3(parameter, ignore_status=True).numel()
            all_size += size
            if parameter.requires_grad:
                tuned_size += size
                tuned_names.append(name)
        return all_size, tuned_size, tuned_names
