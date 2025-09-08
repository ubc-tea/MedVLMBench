import abc
import torch.nn as nn
from easydict import EasyDict as edict
import copy
from utils.utils import maybe_zero_3


class BaseModel:
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.name = ""
        self.model_type = ""

        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.image_processor_callable = None

        self.constants = edict()

    def load_from_pretrained(self, model_path, **kwargs):
        pass

    def load_for_training(self, model_path):
        pass

    @abc.abstractmethod
    def save(self, output_folder, trainer=None):
        pass

    def get_parameters_info(self):
        tuned_parameters = []
        tuned_parameter_size = 0
        all_parameter_size = 0
        
        for n, p in self.model.named_parameters():
            all_parameter_size += maybe_zero_3(p, ignore_status=True).numel()
            if p.requires_grad:
                tuned_parameters.append(n)
                tuned_parameter_size += maybe_zero_3(p, ignore_status=True).numel()

        return all_parameter_size, tuned_parameter_size, tuned_parameters
