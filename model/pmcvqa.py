import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from transformers import LlamaTokenizer
from chat import ChatMetaModel
from easydict import EasyDict as edict

from release.pmcvqa.model.qa_model import QA_model


class PMCVQA(ChatMetaModel):
    def __init__(self, args=None):
        super().__init__(args)

        # Default configuration values from the PMCVQA code snippet
        self.model_path = "./LLAMA/llama-7b-hf"
        self.checkpoint_path = "./Results/QA_no_pretrain_no_aug/Slake/checkpoint-16940"
        self.visual_model_path = "./img_checkpoint/PMC-CLIP/checkpoint.pt"
        self.tokenizer_path = "./LLAMA/tokenizer"
        # If needed, we can store Vision_module='PMC-CLIP' or other defaults, but QA_model should handle that.

        # Determine device (if args doesn't specify, default to cuda if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the tokenizer
        # Assuming LLaMA tokenizer is available at self.tokenizer_path
        self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Initialize the QA_model with its default arguments as given in the original PMCVQA code
        # The QA_model internally sets up the model architecture.
        self.model = QA_model(
            model_path=self.model_path,
            ckp=self.checkpoint_path,
            Vision_module="PMC-CLIP",
            visual_model_path=self.visual_model_path,
            # Other default parameters from the original code can be passed if needed,
            # If QA_model uses dataclasses internally, ensure these match the defaults.
        )

        # Load model checkpoint
        ckp_file = self.checkpoint_path + "/pytorch_model.bin"
        ckpt = torch.load(ckp_file, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations similar to what Slake_Dataset might have used.
        # These might need adjusting based on the dataset's original transforms.
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def infer_vision_language(self, image: Image.Image, qs: str, image_size=None):
        """
        Given a PIL image and a question string, return the predicted answer.
        """
        # Tokenize the question
        input_ids = self.tokenizer(qs, return_tensors="pt").to(self.device)["input_ids"]

        # Transform image
        images = self.image_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Generate the answer
            generation_ids = self.model.generate_long_sentence(input_ids, images)

        generated_text = self.tokenizer.decode(generation_ids[0], skip_special_tokens=True).strip()
        return generated_text


if __name__ == "__main__":
    # blip_caption = BLIP(mode="caption")
    blip_vqa = PMCVQA(args=edict(device="cuda"))

    image_path = "/fast/rjin02/DataSets/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"
    image_path = "/fast/rjin02/DataSets/COCO/2014/val2014/COCO_val2014_000000000042.jpg"

    # caption = blip_caption.caption(image_path)
    # print("Generated Caption:", caption)

    image = Image.open(image_path).convert("RGB")

    question = "What is in the image?"
    answer = blip_vqa.infer_vision_language(image, question, image_size=None)
    print("VQA Answer:", answer)
