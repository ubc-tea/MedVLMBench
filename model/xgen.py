import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria

from model.base import BaseModel
from model.chat import ChatMetaModel


def apply_prompt_template(prompt):
    s = (
            '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
            f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
        )
    return s 


class ImageProcessorCallable:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image):
        # TODO: check for batch > 1
        breakpoint()
        self.image_processor(image)
        return self.image_processor(image)["pixel_values"][0]


class XGenMiniV1(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)
        self.name = "XGenMiniV1"
        self.model_name = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        
        self.hf_path = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        self.model = AutoModelForVision2Seq.from_pretrained(self.hf_path, trust_remote_code=True).to(self.args.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, trust_remote_code=True, use_fast=False, legacy=False)
        self.img_processor = AutoImageProcessor.from_pretrained(self.hf_path, trust_remote_code=True)
        # self.image_processor_callable = ImageProcessorCallable(self.img_processor)
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

    def infer_vision_language(self, image, qs, image_size=None):
        """
        Generates an answer based on input image and text prompt.
        
        :param image: The image in PIL format or a tensor convertible to PIL.
        :param qs: The input question/prompt as a string.
        :param image_size: Optional parameter for image size if resizing is needed.
        
        :return: Generated text output.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        else:
            assert image.dim() == 4

        inputs = self.img_processor([to_pil_image(img_tensor) for img_tensor in image], return_tensors="pt")
        prompt = apply_prompt_template(qs)
        language_inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {k: v.to(self.args.device) for k, v in inputs.items()}

        outputs = self.model.generate(**inputs, image_size=[image[0].size],
                                pad_token_id=self.tokenizer.pad_token_id,
                                do_sample=False, max_new_tokens=768, top_p=None, num_beams=1
                                )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|end|>")[0]
        return answer
