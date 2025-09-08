import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import SiglipModel, SiglipProcessor, SiglipImageProcessor, AutoTokenizer
from model.lora_base import LoRALPModel
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel



class SiglipForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs) -> None:
        model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")        
        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)
        
        processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = processor.tokenizer
        image_processor_hf = processor.image_processor

        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor

        self.initialize_prototypes()
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 768
        self.vision_embed_dim = 768
    
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        return self.model.get_text_features(**inputs)
    
    def encode_image(self, images):
        return self.model.get_image_features(images)

    def forward(self, pixel_values, return_loss=False):
        output = self.model.forward(input_ids=self.prototype["input_ids"], pixel_values=pixel_values, return_loss=return_loss)
        return output.logits_per_image


class SiglipLPForDiagnosis(CLIPImgLPModel):
    def __init__(self, args, text, num_classes) -> None:
        super().__init__(text=text, num_classes=num_classes, model=SiglipModel.from_pretrained("google/siglip-base-patch16-224"), args=args)
        
        image_processor_hf = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 768
        self.vision_embed_dim = 768


class SiglipLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs) -> None:
        model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
        vision_model = model.vision_model
        vision_model.feat_dim = 768
        lora_config = LoraConfig(target_modules=["k_proj", "v_proj", "q_proj"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])

        image_processor_hf = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def extract_features(self, images):
        # The feature extraction logic is identical to CLIP's ViT
        return self.encoder(images).pooler_output