# import torch
# import torch.nn as nn
# from collections import OrderedDict
# from model.base import BaseModel
# from easydict import EasyDict as edict
# from transformers import CLIPModel, SiglipModel


# class LPModel(BaseModel, nn.Module):
#     def __init__(self, args, model, num_classes):
#         super().__init__(args)
#         self.num_classes = num_classes
#         self.model_type = None

#         if isinstance(model, CLIPModel):
#             self.model_type = "huggingface-clip"
#             self.img_encoder = model.vision_model
#             self.visual_projection = model.visual_projection
#             self.feat_dim = 768
#         elif isinstance(model, SiglipModel):
#             self.model_type = "huggingface-siglip"
#             self.img_encoder = model.vision_model
#         else:
#             raise ValueError("Unsupported model type for LPModel")
        

#         self.head = torch.nn.Linear(self.feat_dim, self.num_classes)
        
        
#         if self.model_type == "huggingface-clip":
#             # https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/clip/modeling_clip.py#L916
#             for param in self.img_encoder.parameters():
#                 param.requires_grad = False
#             for param in self.visual_projection.parameters():
#                 param.requires_grad = False
        
#         elif self.model_type == "huggingface-siglip":
#             # https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/siglip/modeling_siglip.py#L960
#             for param in self.img_encoder.parameters():
#                 param.requires_grad = False
        
#         elif self.model_type is None:
#             raise ValueError("Model type not set. Please provide a valid model.")
            

#     def extract_features(self, images):
#         if self.model_type == "huggingface-clip":
#             features = self.img_encoder(images).pooler_output
#             return self.visual_projection(features)
#         elif self.model_type == "siglip":
#             return self.model.img_encoder(images).pooler_output
#         return None

#     def forward(self, images):
#         features = self.extract_features(images)
#         return self.head(features)

#     def load_for_training(self, model_path):
#         pass
        
#     def load_from_pretrained(self, model_path, device, **kwargs):
#         # The model's state_dict is now from the nn.Sequential wrapper
#         model_ckpt = torch.load(model_path)
#         self.model.load_state_dict(model_ckpt)
#         self.model.to(device)