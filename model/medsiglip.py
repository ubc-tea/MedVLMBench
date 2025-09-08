import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import (
    SiglipModel,
    SiglipProcessor,
    SiglipImageProcessor,
    SiglipTokenizer,
)
# Make sure these base classes are correctly imported from your project structure
from model.base import BaseModel
from model.lora_base import LoRALPModel
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel


class MedSigLIPForDiagnosis(CLIPBase):
    """
    Wrapper around `google/medsiglip-448` for zero-/few-shot medical
    image classification or retrieval.
    """
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs):
        # Load MedSigLIP checkpoint
        model = SiglipModel.from_pretrained("google/medsiglip-448")
        # if args and getattr(args, "usage", None) == "clipimg-lora":
        #     lora_cfg = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj"])
        #     # Freeze all parameters before applying LoRA
        #     for _, p in model.named_parameters():
        #         p.requires_grad = False
            # model.vision_model = get_peft_model(model.vision_model, lora_cfg)
        
        super().__init__(
            text=text,
            num_classes=num_classes,
            model=model,
            args=args,
            **kwargs,
        )

        processor = SiglipProcessor.from_pretrained("google/medsiglip-448")
        self.tokenizer: SiglipTokenizer = processor.tokenizer
        
        self.image_processor = ImageProcessorCallable(processor.image_processor)
        self.image_processor_evaluation = self.image_processor
        
        self.initialize_prototypes()
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 1152
        self.vision_embed_dim = 1152

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        return self.model.get_text_features(**inputs)

    def encode_image(self, images):
        return self.model.get_image_features(images)

    def forward(self, pixel_values, return_loss=False):
        output = self.model.forward(input_ids=self.prototype["input_ids"], pixel_values=pixel_values, return_loss=return_loss)
        return output.logits_per_image


class MedSigLIPLPForDiagnosis(CLIPImgLPModel):
    _keys_to_ignore_on_save = []
    def __init__(self, args, text, num_classes) -> None:
        super().__init__(text=text, num_classes=num_classes, model=SiglipModel.from_pretrained("google/medsiglip-448"), args=args)
        
        image_processor_hf = SiglipImageProcessor.from_pretrained("google/medsiglip-448")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 1152
        self.vision_embed_dim = 1152


class MedSigLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs):
        model = SiglipModel.from_pretrained("google/medsiglip-448")
        vision_model = model.vision_model
        vision_model.feat_dim = getattr(vision_model.config, "hidden_size", 1024)

        lora_cfg = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj"])
        
        super().__init__(
            args=args,
            lora_config=lora_cfg,
            encoder=vision_model,
            num_classes=kwargs["num_classes"],
        )

        img_proc_hf = SiglipImageProcessor.from_pretrained("google/medsiglip-448")
        self.image_processor = ImageProcessorCallable(image_processor=img_proc_hf)
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        outputs = self.encoder(images)
        return outputs["pooler_output"]