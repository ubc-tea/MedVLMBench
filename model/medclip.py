import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel
from model.lora_base import LoRALPModel


class MedCLIPFeatureExtractor:
    """Custom feature extractor for MedCLIP to match its original preprocessing."""
    def __init__(self, 
                 crop_size=(224, 224),
                 do_center_crop=True,
                 do_convert_rgb=True,
                 do_normalize=True,
                 do_pad_square=True,
                 do_rescale=True,
                 do_resize=True,
                 image_mean=(0, 0, 0),
                 image_std=(0.26862954, 0.26130258, 0.27577711),
                 rescale_factor=0.5862785803043838,
                 size=224):
        transform_list = []
        if do_convert_rgb:
            transform_list.append(transforms.Lambda(lambda img: img.convert('RGB')))
        if do_pad_square:
            transform_list.append(transforms.Lambda(self.pad_to_square))
        if do_resize:
            transform_list.append(transforms.Resize(size))
        if do_center_crop:
            transform_list.append(transforms.CenterCrop(crop_size))
        transform_list.append(transforms.ToTensor())
        if do_rescale:
            transform_list.append(transforms.Lambda(lambda tensor: tensor * rescale_factor))
        if do_normalize:
            transform_list.append(transforms.Normalize(mean=image_mean, std=image_std))
        self.transform = transforms.Compose(transform_list)

    def pad_to_square(self, img):
        max_dim = max(img.size)
        padding = [(max_dim - img.size[0]) // 2, (max_dim - img.size[1]) // 2]
        padding += [max_dim - img.size[0] - padding[0], max_dim - img.size[1] - padding[1]]
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

    def __call__(self, image):
        return self.transform(image)

class MedCLIPForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, args=None, *kargs, **kwargs) -> None:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()

        if args and args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["query", "key", "value"])
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_config)

        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)
    
        self.processor = MedCLIPProcessor()
        self.tokenizer = self.processor.tokenizer
        self.image_processor = ImageProcessorCallable(MedCLIPFeatureExtractor())
        self.image_processor_evaluation = self.image_processor
        
        self.initialize_prototypes()

    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 512
        self.vision_embed_dim = 512
        
    @torch.no_grad()
    def encode_text(self, text):
        assert len(text) == self.num_classes
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        return self.model.encode_text(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    def encode_image(self, images):
        return self.model.encode_image(images)

    def forward(self, pixel_values, return_loss=False):
        output = super().forward(pixel_values=pixel_values, return_loss=return_loss)
        return output.logits_per_image

    def forward(self, pixel_values, return_loss=False):
        outputs = super().forward(pixel_values=pixel_values, return_loss=return_loss)
        return outputs["logits"]



class MedCLIPLPForDiagnosis(CLIPImgLPModel):
    def __init__(self, args, text, num_classes) -> None:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        super().__init__(text=text, num_classes=num_classes, model=model, args=args)
        
        self.image_processor = ImageProcessorCallable(MedCLIPFeatureExtractor())
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model
        self.text_embed_dim = 512
        self.vision_embed_dim = 512

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, text):
        return self.model.encode_text(text)
    

class MedCLIPLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs) -> None:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        vision_model = model.vision_model
        vision_model.feat_dim = 512
        lora_config = LoraConfig(target_modules=["qkv"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])
        
        self.image_processor = ImageProcessorCallable(MedCLIPFeatureExtractor())
        self.image_processor_evaluation = self.image_processor
    
    def extract_features(self, images):
        # MedCLIP vision encoder returns a tensor directly
        return self.encoder(images)