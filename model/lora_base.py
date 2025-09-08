import torch
import torch.nn as nn
import copy
from collections import OrderedDict
from model.base import BaseModel
from easydict import EasyDict as edict
from peft import LoraConfig, get_peft_model


class LoRALPModel(BaseModel, nn.Module):
    def __init__(self, args, lora_config, encoder, num_classes):
        super().__init__(args)
        self.num_classes = num_classes
        self.lora_config = lora_config
        # Apply LoRA to a deepcopy of the encoder
        self.encoder = get_peft_model(copy.deepcopy(encoder), lora_config)
        self.head = torch.nn.Linear(encoder.feat_dim, num_classes)
        
        # Combine encoder and head into a single model attribute for consistent interface
        self.model = nn.Sequential(
            OrderedDict(
                    [
                        ("encoder", self.encoder),
                        ("head", self.head)
                    ]
                )
        )
    
    def extract_features(self, images):
        """
        Subclasses must implement this method to extract features from the encoder's output.
        """
        raise NotImplementedError

    def forward(self, images):
        features = self.extract_features(images)
        return self.head(features)

    def load_for_training(self, model_path):
        pass
        
    def load_from_pretrained(self, model_path, device, **kwargs):
        # The model's state_dict is now from the nn.Sequential wrapper
        model_ckpt = torch.load(model_path)
        self.model.load_state_dict(model_ckpt)
        self.model.to(device)