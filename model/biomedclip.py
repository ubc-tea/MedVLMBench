import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model
from open_clip import create_model_from_pretrained, get_tokenizer
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel
from model.lora_base import LoRALPModel


class BiomedCLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs) -> None:
        model, processor = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )

        if args and args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["qkv"])
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.visual = get_peft_model(model.visual, lora_config)
        
        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)
        
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.image_processor = ImageProcessorCallable(processor)
        self.image_processor_evaluation = self.image_processor
        
        # This model has a built-in logit_scale
        self.logit_scale = self.model.logit_scale
        
        self.initialize_prototypes()
    
    def setup_encoders(self):
        self.model.vision_model = self.model.visual
        self.model.text_model = self.model.text
        self.text_embed_dim = 512
        self.vision_embed_dim = 512

    @torch.no_grad()
    def encode_text(self, text):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, context_length=256).to(device)
        return self.model.encode_text(inputs, normalize=False)
    
    def encode_image(self, images):
        return self.model.encode_image(images)
    
    def initialize_prototypes(self):
        """Initializes text prototypes. Should be called at the end of subclass __init__."""
        if self.prototype is None:
            self.prototype = self.tokenizer(self.prototype_text)
            self.prototype = self.prototype.to(self.args.device)
    
    def forward(self, pixel_values):
        image_features, text_features, logit_scale = self.model(pixel_values, self.prototype)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        return logits


class BioMedCLIPLPForDiagnosis(CLIPImgLPModel):
    def __init__(self, args, text, num_classes) -> None:
        model, preprocess = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )  
        super().__init__(text=text, num_classes=num_classes, model=model, args=args)
        
        self.image_processor = ImageProcessorCallable(preprocess)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.model.vision_model = self.model.visual
        self.model.text_model = self.model.text
        self.text_embed_dim = 512
        self.vision_embed_dim = 512

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, text):
        return self.model.encode_text(text)

    # def forward(self, pixel_values, input_ids):
    #     image_features, text_features, logit_scale = self.model.forward(pixel_values, input_ids)
    #     logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    #     return logits



class BioMedCLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs) -> None:
        model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )  
        vision_model = model.visual
        vision_model.feat_dim = 512
        lora_config = LoraConfig(target_modules=["qkv"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])

        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        self.image_processor = ImageProcessorCallable(transform)
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        return self.encoder(images, out_type='raw')[:, 0]