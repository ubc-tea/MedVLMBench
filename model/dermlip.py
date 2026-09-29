import os

from open_clip import create_model_and_transforms, create_model_from_pretrained, get_tokenizer
from model.openclip_base import OpenCLIPForDiagnosis, OpenCLIPLPForDiagnosis

_DERMLIP_REPO = "hf-hub:redlessone/DermLIP_ViT-B-16"
_DERMLIP_LOCAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pretrained_models", "dermlip", "frozen_lp_encoder.pt",
)


class _DermLIPMixin:
    def build_model(self):
        if os.path.isfile(_DERMLIP_LOCAL):
            model, _, preprocess = create_model_and_transforms(
                "ViT-B-16", pretrained=_DERMLIP_LOCAL
            )
            return model, preprocess
        return create_model_from_pretrained(_DERMLIP_REPO)

    def tokenize(self, texts):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = get_tokenizer(
                "ViT-B-16" if os.path.isfile(_DERMLIP_LOCAL) else _DERMLIP_REPO
            )
        return self._tokenizer(texts)


class DermLIPForDiagnosis(_DermLIPMixin, OpenCLIPForDiagnosis):
    pass


class DermLIPLPForDiagnosis(_DermLIPMixin, OpenCLIPLPForDiagnosis):
    pass
