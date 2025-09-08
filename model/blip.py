import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from transformers import AutoTokenizer
from transformers import BlipProcessor, BlipImageProcessor, BlipConfig, BlipForConditionalGeneration, BlipForQuestionAnswering, BlipModel
from peft import LoraConfig, get_peft_model

from model.base import BaseModel
from model.chat import ChatMetaModel
from model.clip_base import CLIPBase, ImageProcessorCallable
from model.lp_base import LPModel
from model.lora_base import LoRALPModel


class BLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs) -> None:
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")

        if args and args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["qkv"])
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_config)

        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
        image_processor_hf = BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
                
        self.initialize_prototypes()
    
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        text_outputs = self.model.text_model(input_ids=text_inputs.input_ids, attention_mask=text_inputs.attention_mask)
        # BLIP does not have a projection head, we use the CLS token output
        return text_outputs.last_hidden_state[:, 0, :]
    
    def encode_image(self, images):
        image_features = self.model.vision_model(images)
        # Use CLS token output as image feature
        return image_features.last_hidden_state[:, 0, :]


class BLIPForQA(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "BLIP"
        self.model_type = "general"
        self.model_name = "Salesforce/blip-vqa-base"
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_name).to(self.args.device)
        
        self.image_processor_callable = ImageProcessorCallable(self.processor.image_processor)
        self.tokenizer = self.processor.tokenizer

    def infer_vision_language(self, image, qs, image_size=None):
        text_inputs = self.tokenizer(qs, return_tensors="pt", padding=True, truncation=True)
        inputs = {
            "input_ids": text_inputs["input_ids"].to(self.args.device),
            "attention_mask": text_inputs["attention_mask"].to(self.args.device),
            "pixel_values": image.to(self.args.device),
        }
        outputs = self.model.generate(**inputs, max_length=768)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return answer


class BLIPLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None:
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        vision_model = model.vision_model
        vision_model.feat_dim = 768
        super().__init__(encoder=vision_model, *args, **kwargs)

        image_processor_hf = BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"][:, 0, :]


class BLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs) -> None:
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        vision_model = model.vision_model
        vision_model.feat_dim = 768
        lora_config = LoraConfig(target_modules=["qkv"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])
        
        image_processor_hf = BlipImageProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.image_processor = ImageProcessorCallable(image_processor_hf)
        self.image_processor_evaluation = self.image_processor
    
    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"][:, 0, :]