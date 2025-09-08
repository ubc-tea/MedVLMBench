import torch
from model.release.xraygpt.models.mini_gpt4 import MiniGPT4
from model.release.xraygpt.processors.blip_processors import Blip2ImageEvalProcessor
from model.chat import ChatMetaModel
from model.lp_base import LPModel
from model.lora_base import LoRALPModel
from model.clip_base import ImageProcessorCallable
from peft import LoraConfig


class XrayGPT(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "XrayGPT-mini"
        self.model_type = "medical"
        self.model = MiniGPT4(
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp32",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model='./pretrained_models/Vicuna_Radiology_fp16/',
            prompt_path='./model/release/xraygpt/prompts/alignment.txt',
            prompt_template='###Patient: {} ###Doctor: ',
            max_txt_len=160,
            low_resource=True,
            end_sym="###",
        )
        ckpt = torch.load("./pretrained_models/xraygpt_pretrained1.pth", map_location="cpu")
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.model.to(self.args.device)
        self.processor = Blip2ImageEvalProcessor()
        self.image_processor_callable = ImageProcessorCallable(self.processor)

    def infer_vision_language(self, image, qs, image_size=None):
        # image is already pre-processed by the callable
        img_embeds, _ = self.model.encode_img(image.to(self.args.device))
        prompt = f"###Patient: <Img><ImageHere></Img> {qs}###Doctor:"
        img_embeds, _ = self.model.prompt_wrap(img_embeds, None, prompt)
        bos = torch.ones((img_embeds.size(0), 1), dtype=torch.long, device=self.args.device) * self.model.llama_tokenizer.bos_token_id
        bos_embeds = self.model.llama_model.model.embed_tokens(bos)
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=self.args.device)
        outputs = self.model.llama_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300,
            num_beams=1, do_sample=True, top_p=0.9, repetition_penalty=1.0, length_penalty=1,
            temperature=1.0, eos_token_id=self.model.llama_tokenizer.eos_token_id,
            pad_token_id=self.model.llama_tokenizer.eos_token_id
        )
        output_token = outputs[0]
        if output_token[0] == 0: output_token = output_token[1:]
        if output_token[0] == 1: output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        return output_text.split('###')[0].split('Doctor:')[-1].strip()

class XGenGPTLPForDiagnosis(LPModel):
    def __init__(self, *args, **kwargs) -> None:
        model = MiniGPT4(vit_model="eva_clip_g", q_former_model="...", llama_model='./pretrained_models/Vicuna_Radiology_fp16/')
        ckpt = torch.load("./pretrained_models/xraygpt_pretrained1.pth", map_location="cpu")
        model.load_state_dict(ckpt['model'], strict=False)
        
        vision_model = model.visual_encoder
        vision_model.feat_dim = 1408
        super().__init__(encoder=vision_model, *args, **kwargs)
        
        self.image_processor = ImageProcessorCallable(Blip2ImageEvalProcessor())
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        # The visual_encoder in MiniGPT4/XrayGPT returns features directly
        # and we take the CLS token representation.
        return self.encoder(images)[:, 0, :]


class XGenGPTLoRALPForDiagnosis(LoRALPModel):
    def __init__(self, args, *kargs, **kwargs) -> None:
        model = MiniGPT4(vit_model="eva_clip_g", q_former_model="...", llama_model='./pretrained_models/Vicuna_Radiology_fp16/')
        ckpt = torch.load("./pretrained_models/xraygpt_pretrained1.pth", map_location="cpu")
        model.load_state_dict(ckpt['model'], strict=False)
        
        vision_model = model.visual_encoder
        vision_model.feat_dim = 1408
        lora_config = LoraConfig(target_modules=["qkv"])
        super().__init__(args=args, lora_config=lora_config, encoder=vision_model, num_classes=kwargs['num_classes'])
        
        self.image_processor = ImageProcessorCallable(Blip2ImageEvalProcessor())
        self.image_processor_evaluation = self.image_processor

    def extract_features(self, images):
        return self.encoder(images)[:, 0, :]