import os
import sys

try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
except ModuleNotFoundError as exc:
    if exc.name != "conch":
        raise
    # Use the pinned official source vendored for reproducible offline runs.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "CONCH"))
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from model.openclip_base import OpenCLIPForDiagnosis, OpenCLIPLPForDiagnosis

# CONCH (Lu et al., Nature Medicine 2024). The weights are gated on Hugging Face: accept the license at
# https://huggingface.co/MahmoodLab/CONCH, then log in (`huggingface-cli login`) or set HF_TOKEN.
# A local copy at pretrained_models/conch/pytorch_model.bin (or $CONCH_CKPT) takes precedence.
_CFG = "conch_ViT-B-16"
_HF_REPO = "hf_hub:MahmoodLab/CONCH"
_LOCAL_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pretrained_models", "conch", "pytorch_model.bin"
)


class _CONCHMixin:
    def build_model(self):
        local = os.environ.get("CONCH_CKPT", _LOCAL_CKPT)
        if os.path.isfile(local):
            return create_model_from_pretrained(_CFG, checkpoint_path=local)
        return create_model_from_pretrained(_CFG, checkpoint_path=_HF_REPO, hf_auth_token=os.environ.get("HF_TOKEN"))

    def tokenize(self, texts):
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = get_tokenizer()
        return tokenize(self._tokenizer, texts)

    # CoCa normalizes by default; keep raw projected embeddings to match the other CLIP wrappers
    def _image_features(self, images):
        return self.model.encode_image(images, proj_contrast=True, normalize=False)

    def _text_features(self, tokens):
        return self.model.encode_text(tokens, normalize=False)


class CONCHForDiagnosis(_CONCHMixin, OpenCLIPForDiagnosis):
    pass


class CONCHLPForDiagnosis(_CONCHMixin, OpenCLIPLPForDiagnosis):
    pass
