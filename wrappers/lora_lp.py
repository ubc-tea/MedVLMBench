import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.base import BaseWrapper
from peft import LoftQConfig, LoraConfig, get_peft_model

class LoraLPWrapper(BaseWrapper):
    def __init__(self, model, num_classes, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)

        self.encoder = model
        self.head = torch.nn.Linear(self.encoder.feat_dim, num_classes)
        self.encoder = copy.deepcopy(get_peft_model(self.encoder, lora_config))

    def forward(self, x):
        return self.head(self.encoder(x))