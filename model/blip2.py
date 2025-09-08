import torch
import torch.nn.functional as F
from transformers import Blip2ForConditionalGeneration, Blip2Processor, Blip2ForImageTextRetrieval
from peft import LoraConfig, get_peft_model
from model.chat import ChatMetaModel
from model.lora_base import LoRALPModel
from model.clip_base import CLIPBase, ImageProcessorCallable, CLIPImgLPModel


class BLIP2(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "BLIP2-2.7b"
        self.model_type = "general"
        self.model_name = "Salesforce/blip2-opt-2.7b"
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name).to(self.args.device)

    def infer_vision_language(self, image, qs, image_size=None):
        qs = "Question: {} Answer:".format(qs)
        inputs = self.processor(images=image, text=qs, return_tensors="pt", padding=True, truncation=True).to(
            self.args.device
        )
        outputs = self.model.generate(**inputs, max_new_tokens=768)
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        return answer


class BLIP2ForDiagnosis(CLIPBase):
    def __init__(self, text, num_classes, args=None, model_name="Salesforce/blip2-itm-vit-g", *kargs, **kwargs):
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name)

        if args and args.usage == "clip-img-lora":
            lora_config = LoraConfig(target_modules=["qkv"]) # qkv for image encoder and ("query", "key", "value") for qformer
            for name, para in model.named_parameters():
                para.requires_grad = False
            model.vision_model = get_peft_model(model.vision_model, lora_config)

        super().__init__(text=text, num_classes=num_classes, model=model, args=args, **kwargs)
        
        processor = Blip2Processor.from_pretrained(model_name)
        self.tokenizer = processor.tokenizer
        self.image_processor = ImageProcessorCallable(processor.image_processor)
        self.image_processor_evaluation = self.image_processor

        self.initialize_prototypes()
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.embeddings
        self.text_embed_dim = 1408
        self.vision_embed_dim = 1408

    def forward(self, pixel_values):
        return self.model.forward(input_ids=self.prototype["input_ids"], attention_mask=self.prototype["attention_mask"], pixel_values=pixel_values).logits_per_image


class BLIP2LPForDiagnosis(CLIPImgLPModel):
    def __init__(self, args, text, num_classes) -> None:
        super().__init__(
                text=text,
                num_classes=num_classes,
                model=Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g"),
                args=args
            )

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-itm-vit-g")
        self.image_processor = ImageProcessorCallable(processor.image_processor)
        self.image_processor_evaluation = self.image_processor
    
    def setup_encoders(self):
        self.vision_model = self.model.vision_model
        self.text_model = self.model.embeddings
        self.text_embed_dim = 1408
        self.vision_embed_dim = 1408

    def encode_image(self, images):
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L2376
        return self.vision_model(images)[0][:, 0, :]


class BLIP2LPLoRAForDiagnosis(LoRALPModel):
    def __init__(self, args, model_name="Salesforce/blip2-itm-vit-g", *kargs, **kwargs) -> None:
        model = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        vision_model = model.vision_model
        vision_model.feat_dim = 1408
        lora_config = LoraConfig(target_modules=["qkv"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])

        processor = Blip2Processor.from_pretrained(model_name)
        self.image_processor = ImageProcessorCallable(processor.image_processor)
        self.image_processor_evaluation = self.image_processor
    
    def extract_features(self, images):
        return self.encoder(images)["last_hidden_state"][:, 0, :]