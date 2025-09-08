import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor, CLIPFeatureExtractor, CLIPTokenizer
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel, CLIPVisionLoRALPModel


class CLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")        
        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        image_processor_hf = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor

        self.initialize_prototypes()
    
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        return self.model.get_text_features(**inputs)
    
    def encode_image(self, images):
        return self.model.get_image_features(images)

    def forward(self, pixel_values, return_loss=False):
        output = super().forward(pixel_values=pixel_values, return_loss=return_loss)
        return output.logits_per_image


class CLIPLPForDiagnosis(CLIPImgLPModel):
    def __init__(self, args, text, num_classes) -> None:
        super().__init__(text=text, num_classes=num_classes, model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), args=args)
        
        image_processor_hf = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 512
        self.vision_embed_dim = 512


class CLIPVisionLoRALPForDiagnosis(CLIPVisionLoRALPModel):
    def __init__(self, args, *kargs, **kwargs) -> None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        lora_config = LoraConfig(target_modules=["k_proj", "v_proj", "q_proj"])
        breakpoint()
        super().__init__(args=args, lora_config=lora_config, text=args.text, num_classes=kwargs['num_classes'])

        image_processor_hf = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 512
        self.vision_embed_dim = 512