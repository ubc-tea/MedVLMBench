from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    CLIPModel,
    CLIPTokenizer,
    CLIPFeatureExtractor,
)

from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel
from model.lora_base import LoRALPModel

__all__ = [
    "PubMedCLIPForDiagnosis",
    "PubMedCLIPLPForDiagnosis",
    "PubMedCLIPLoRALPForDiagnosis",
]


_PUBMED_CLIP_REPO = "flaviagiammarino/pubmed-clip-vit-base-patch32"


class PubMedCLIPForDiagnosis(CLIPBase):
    """Zero-/few-shot classification wrapper around PubMedCLIP.

    Parameters
    ----------
    text : list[str]
        A *list* of class-name prompts. Length must equal *num_classes*.
    num_classes : int
        Number of diagnosis classes.
    args : argparse.Namespace | None, optional
        Configuration namespace produced by `parse_args()`. If provided and
        ``args.usage == "pubmedclip-img-lora"`` the vision encoder is wrapped
        with a LoRA adapter while all original parameters are frozen.
    **kwargs
        Forwarded to :class:`model.clip_base.CLIPBase`.
    """

    def __init__(self, text, num_classes, args=None, *kargs, **kwargs):
        model = CLIPModel.from_pretrained(_PUBMED_CLIP_REPO)

        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)

        self.tokenizer = CLIPTokenizer.from_pretrained(_PUBMED_CLIP_REPO)
        _feature_extractor = CLIPFeatureExtractor.from_pretrained(_PUBMED_CLIP_REPO)
        self.image_processor = ImageProcessorCallable(_feature_extractor)
        self.image_processor_evaluation = self.image_processor

        self.initialize_prototypes()
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 512
        self.vision_embed_dim = 512

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes, "#text prompts must equal num_classes"
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        return self.model.get_text_features(**inputs)

    def encode_image(self, images):
        return self.model.get_image_features(images)

    def forward(self, pixel_values, return_loss=True):
        output = super().forward(pixel_values=pixel_values, return_loss=return_loss)
        
        return output.logits_per_image

    

class PubMedCLIPLPForDiagnosis(CLIPImgLPModel):
    def __init__(self, args, text, num_classes) -> None:
        super().__init__(text=text, num_classes=num_classes, model=CLIPModel.from_pretrained(_PUBMED_CLIP_REPO), args=args)
        
        image_processor_hf = CLIPFeatureExtractor.from_pretrained(_PUBMED_CLIP_REPO)
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 512
        self.vision_embed_dim = 512


class PubMedCLIPLoRALPForDiagnosis(LoRALPModel):
    """LoRA-based parameter-efficient fine-tuning for PubMedCLIP."""

    def __init__(self, args, *kargs, **kwargs):
        model = CLIPModel.from_pretrained(_PUBMED_CLIP_REPO)
        vision_model = model.vision_model
        vision_model.feat_dim = 768

        lora_cfg = LoraConfig(target_modules=["k_proj", "v_proj", "q_proj"])
        breakpoint()
        super().__init__(args=args, lora_config=lora_cfg, encoder=vision_model, num_classes=kwargs["num_classes"])

        _feature_extractor = CLIPFeatureExtractor.from_pretrained(_PUBMED_CLIP_REPO)
        self.image_processor = ImageProcessorCallable(_feature_extractor)
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"][:, 0, :]
