import torch
import torch.nn as nn
import torch.nn.functional as F
import clip  # Assumes the OpenAI CLIP package is installed
from model.clip_base import CLIPBase


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)
    

class GenericTextEncoder(nn.Module):
    def __init__(self, classnames, clip_model, template="a photo of a {}"):
        super().__init__()
        self.classnames = classnames
        self.clip_model = clip_model
        self.template = template
        self.dtype = clip_model.dtype
    
    def forward(self):
        # Build prompts using the given template.
        prompts = [self.template.format(c.replace('_', ' ')) for c in self.classnames]
        # Tokenize and move tokens to the same device as the clip model.
        tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(next(self.clip_model.parameters()).device)
        text_features = self.clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

class GenericCLIPAdapter(nn.Module):
    def __init__(self, clip_model, classnames, adapter_in_dim=None, reduction=4, template="a photo of a {}"):
        super().__init__()
        self.clip_model = clip_model
        self.image_encoder = clip_model.visual
        self.text_encoder = GenericTextEncoder(classnames, clip_model, template)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # If not provided, assume a default input dimension (e.g. 1024 for ViT-B/32)
        if adapter_in_dim is None:
            adapter_in_dim = 1024
        self.adapter = Adapter(adapter_in_dim, reduction).to(self.dtype)
        # A ratio to blend the adapted features with the original ones.
        self.ratio = 0.2

    def forward(self, image):
        # Compute image features.
        image_features = self.image_encoder(image.type(self.dtype))
        # Pass features through the adapter.
        adapted = self.adapter(image_features)
        # Blend the adapted and original features.
        image_features = self.ratio * adapted + (1 - self.ratio) * image_features
        # Normalize features.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # Get the text features (computed on the fly).
        text_features = self.text_encoder()
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

class CLIPAdapterWrapper(CLIPBase):
    def __init__(self, text, num_classes, clip_model, adapter_in_dim=None, reduction=4, template="a photo of a {}"):
        """
        text: list of class names (strings)
        num_classes: number of classes (usually len(text))
        clip_model: a pre-loaded CLIP model instance
        adapter_in_dim: dimension of image features (if None, a default is used)
        reduction: reduction factor in the Adapter
        template: template to build text prompts
        """
        # First, initialize the parent class. Use a temporary placeholder (None) for the model.
        super().__init__(text, num_classes, model=None)
        # Now assign module attributes after nn.Module is initialized.
        self.clip_model = clip_model
        self.template = template
        self.model = GenericCLIPAdapter(clip_model, text, adapter_in_dim, reduction, template)
        
        # Freeze all parameters except for the adapter.
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
        
        # Precompute and store the text (prototype) features.
        self.prototype = self.encode_text(text)
    
    def forward(self, images):
        # Forward pass returns the logits.
        logits = self.model(images)
        return logits
    
    def encode_img(self, images):
        # Compute image features similarly to forward(), but return embeddings.
        image_features = self.clip_model.visual(images.type(self.clip_model.dtype))
        adapted = self.model.adapter(image_features)
        image_features = self.model.ratio * adapted + (1 - self.model.ratio) * image_features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    @torch.no_grad()
    def encode_text(self, text):
        # Given a list of class names, build prompts and compute text features.
        prompts = [self.template.format(c.replace('_', ' ')) for c in text]
        tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(next(self.clip_model.parameters()).device)
        text_features = self.clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def load_for_training(self, model_path):
        """
        Loads the adapter (only) weights from the checkpoint.
        The checkpoint is assumed to contain the adapter's state_dict.
        """
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.model.adapter.load_state_dict(state_dict, strict=False)
        print("Loaded adapter weights from", model_path)
    
    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)