import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel, CLIPProcessor, CLIPFeatureExtractor
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel
from model.lora_base import LoRALPModel


class PLIPForDiagnosis(CLIPBase):
    """
    Full-model fine-tuning or LoRA-tuning wrapper around the
    ViNID/PLIP CLIP checkpoint for zero-/few-shot diagnosis tasks.
    """
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs):
        # Load base PLIP model
        model = CLIPModel.from_pretrained("vinid/plip")
        super().__init__(text=text,
                         num_classes=num_classes,
                         model=model,
                         args=args,
                         **kwargs)

        # Shared tokenizer / image processor from CLIPProcessor
        processor = CLIPProcessor.from_pretrained("vinid/plip")
        self.tokenizer = processor.tokenizer

        self.image_processor = ImageProcessorCallable(processor.image_processor)
        self.image_processor_evaluation = self.image_processor

        self.initialize_prototypes()

    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        return self.model.get_text_features(**inputs)

    def encode_image(self, images):
        return self.model.get_image_features(images)
    
    def forward(self, pixel_values, return_loss=False):
        return super().forward(pixel_values, return_loss).logits_per_image


class PLIPLPForDiagnosis(CLIPImgLPModel):
    def __init__(self, args, text, num_classes) -> None:
        super().__init__(text=text, num_classes=num_classes, model=CLIPModel.from_pretrained("vinid/plip"), args=args)
        
        image_processor_hf = CLIPFeatureExtractor.from_pretrained("vinid/plip")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 512
        self.vision_embed_dim = 512


class PLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs):
        model = CLIPModel.from_pretrained("vinid/plip")
        vision_model = model.vision_model
        vision_model.feat_dim = getattr(vision_model, "feat_dim", 768)

        lora_config = LoraConfig(target_modules=["k_proj", "q_proj", "v_proj"])
        breakpoint()
        super().__init__(args=args,
                         lora_config=lora_config,
                         encoder=vision_model,
                         num_classes=kwargs["num_classes"])

        image_processor_hf = CLIPFeatureExtractor.from_pretrained("vinid/plip")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"][:, 0, :]
