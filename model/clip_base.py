import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torchvision.transforms.functional import to_pil_image
from model.base import BaseModel
from easydict import EasyDict as edict
from utils.utils import maybe_zero_3


class ImageProcessorCallable:
    """
    A callable wrapper for image processors to handle batches of tensors.
    It converts tensors to PIL images, applies the processor, and stacks the results.
    """
    def __init__(self, image_processor, transform_func=None):
        self.image_processor = image_processor
        if transform_func is None:
            self.transform_func = lambda x: x
        else:
            self.transform_func = transform_func

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        try:
            processed = self.image_processor(image, return_tensors="pt")
        except TypeError as e:
            if "got an unexpected keyword argument 'return_tensors'" in str(e):
                processed = self.image_processor(image)
            else:
                raise

        if isinstance(processed, torch.Tensor):
            if processed.dim() == 3:
                return processed

        if 'pixel_values' not in processed:
            raise ValueError("The image processor must return 'pixel_values' in the processed output.")
        
        if processed['pixel_values'].dim() == 3:
            return processed['pixel_values']
        elif processed['pixel_values'].dim() == 4:
            return processed['pixel_values'].squeeze(0)
        
        return None


class CLIPBase(BaseModel, nn.Module):
    def __init__(self, text, num_classes, model=None, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.model = model
        self.prototype = None
        self.prototype_text = text
        self.num_classes = num_classes
        self.setup_encoders()

        if self.args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["k_proj", "v_proj", "q_proj"])
            for name, para in self.model.named_parameters():
                para.requires_grad = False
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=self.find_target_linear_names(self.model.vision_model),
            )
            self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)
            self.tokenizer = None

    def initialize_prototypes(self):
        """Initializes text prototypes. Should be called at the end of subclass __init__."""
        if self.prototype is None:
            self.prototype = self.tokenizer(self.prototype_text, padding=True, return_tensors="pt")
            self.prototype = self.prototype.to(self.args.device)
    
    def setup_encoders(self):
        self.text_embed_dim = self.model.text_embed_dim
        self.vision_embed_dim = self.model.vision_embed_dim

    def forward(self, pixel_values, return_loss=True):
        output = self.model.forward(input_ids=self.prototype["input_ids"], attention_mask=self.prototype["attention_mask"], pixel_values=pixel_values, return_loss=return_loss)
        return output

    def encode_image(self, images):
        return self.model.get_image_features(images)

    def encode_text(self, text):
        return self.model.get_text_features(text)

    def load_for_training(self, model_path):
        pass

    def load_from_pretrained(self, model_path, device, **kwargs):
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)
        
    def find_target_linear_names(self, model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
            linear_cls = torch.nn.Linear
            embedding_cls = torch.nn.Embedding
            lora_module_names = []
            for name, module in model.named_modules():
                if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
                    continue
                if isinstance(module, (linear_cls, embedding_cls)):
                    lora_module_names.append(name)
            if num_lora_modules > 0:
                lora_module_names = lora_module_names[-num_lora_modules:]
            if verbose:
                self.args.logger.info(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
            return lora_module_names
    

class CLIPImgLPModel(CLIPBase):
    def __init__(self, text, num_classes, model=None, args=None):
        super().__init__(text, num_classes, model, args)
        self.num_classes = num_classes

        assert args.usage in ["lp", "img-lora-lp"], f"Unsupported usage: {args.usage}"
        for param in self.model.parameters():
            param.requires_grad = False

        if args.usage == "img-lora-lp":
            self.args.logger.info("Using LoRA for CLIP image model")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=self.find_target_linear_names(self.model.vision_model),
            )
            self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)
        
        self.head = torch.nn.Linear(self.vision_embed_dim, self.num_classes)


    def forward(self, images):
        with torch.no_grad():
            image_features = self.encode_image(images)
        return self.head(image_features)

    def get_parameters_info(self):
        tuned_parameters = []
        tuned_parameter_size = 0
        all_parameter_size = 0
        
        for n, p in self.named_parameters():
            all_parameter_size += maybe_zero_3(p, ignore_status=True).numel()
            if p.requires_grad:
                tuned_parameters.append(n)
                tuned_parameter_size += maybe_zero_3(p, ignore_status=True).numel()

        return all_parameter_size, tuned_parameter_size, tuned_parameters



class CLIPVisionLoRALPModel(BaseModel, nn.Module):
    def __init__(self, args, model, num_classes, lora_config):
        super().__init__(args)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.vision_model = get_peft_model(self.model.vision_model, lora_config)
        
        self.head = nn.Linear(self.feat_dim, self.num_classes).to(self.device)
        

    def forward(self, images):
        image_features = self.encode_image(images)
        
        return self.head(image_features)        