from dataset.utils import get_prototype


def get_model(args, **kwargs):
    if args.task == "vqa" or args.task == "caption":
        if args.model == "BLIP":
            from model.blip import BLIPForQA

            model = BLIPForQA(args=args)
        elif args.model == "LLaVA-1.5":
            from model.llava import LLaVA

            model = LLaVA(args=args)
        elif args.model == "BLIP2-2.7b":
            from model.blip2 import BLIP2

            model = BLIP2(args=args)
        elif args.model == "LLaVA-Med":
            from model.llava_med import LLaVAMed

            model = LLaVAMed(args=args)
        elif args.model == "Gemma3":
            from model.gemma3 import Gemma3

            model = Gemma3(args=args)
        elif args.model == "InternVL3":
            from model.internvl import InternVL3

            model = InternVL3(args=args)
        elif args.model == "MedGemma":
            from model.medgemma import MedGemma

            model = MedGemma(args=args)
        elif args.model == "Qwen2-VL":
            from model.qwen2_vl import Qwen2_VL

            model = Qwen2_VL(args=args)
        elif args.model == "Qwen25-VL":
            from model.qwen25_vl import Qwen25_VL

            model = Qwen25_VL(args=args)
        elif args.model == "XGenMiniV1":
            from model.xgen import XGenMiniV1

            model = XGenMiniV1(args=args)
        elif args.model == "XrayGPT":
            from model.xraygpt import XrayGPT

            model = XrayGPT(args=args)
        elif args.model in ["NVILA", "VILA-M3", "VILA1.5"]:
            from model.vila import VILA

            model = VILA(args=args)
        elif args.model == "Lingshu":
            from model.lingshu import Lingshu

            model = Lingshu(args=args)
        elif args.model == "o3":
            from model.gpt import o3

            model = o3(args=args)
        elif args.model == "gemini-2.5-pro":
            from model.gemini import Gemini25Pro

            model = Gemini25Pro(args=args)
        else:
            raise NotImplementedError()
    
    elif args.task == "caption":
        if args.model == "BLIP":
            model = BLIPForQA(args=args)
        elif args.model == "LLaVA-1.5":
            model = LLaVA(args=args)
        elif args.model == "BLIP2-2.7b":
            model = BLIP2(args=args)
        elif args.model == "LLaVA-Med":
            model = LLaVAMed(args=args)
        elif args.model == "XGenMiniV1":
            model = XGenMiniV1(args=args)
        elif args.model == "XrayGPT":
            from model.xraygpt import XrayGPT

            model = XrayGPT(args=args)
        else:
            raise NotImplementedError()

    elif args.task == "diagnosis":
        from dataset.diagnosis import INFO
        text = get_prototype(args)
        text = ["a photo of {}".format(txt) for txt in text]
        num_classes = len(INFO[args.dataset.lower()]["label"])

        if args.usage in ["lp", "img-lora-lp"]:
            if args.model == "BLIP":
                from model.blip import BLIPLPForDiagnosis

                model = BLIPLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "XrayGPT":
                from model.xraygpt import XGenGPTLPForDiagnosis

                model = XGenGPTLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                from model.biomedclip import BioMedCLIPLPForDiagnosis

                model = BioMedCLIPLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "CLIP":
                from model.clip import CLIPLPForDiagnosis

                model = CLIPLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "MedCLIP":
                from model.medclip import MedCLIPLPForDiagnosis

                model = MedCLIPLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BLIP2-2.7b":
                from model.blip2 import BLIP2LPForDiagnosis

                model = BLIP2LPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PMCCLIP":
                from model.pmcclip import PMCCLIPForDiagnosis, PMCCLIPLPForDiagnosis

                model = PMCCLIPLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PLIP":
                from model.plip import PLIPLPForDiagnosis

                model = PLIPLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "MedSigLIP":
                from model.medsiglip import MedSigLIPLPForDiagnosis

                model = MedSigLIPLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PubMedCLIP":
                from model.pubmedclip import PubMedCLIPLPForDiagnosis
                
                model = PubMedCLIPLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "SigLIP":
                from model.siglip import SiglipLPForDiagnosis

                model = SiglipLPForDiagnosis(args=args, text=text, num_classes=num_classes)
            else:
                raise NotImplementedError()
            
        elif args.usage == "lora_lp":
            raise NotImplementedError("LoRA-LP is not supported for diagnosis task.")
            if args.model == "BLIP":
                from model.blip import BLIPLoRALPForDiagnosis

                model = BLIPLoRALPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "XrayGPT":
                from model.xraygpt import XGenGPTLoRALPForDiagnosis

                model = XGenGPTLoRALPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                from model.biomedclip import BioMedCLIPLoRALPForDiagnosis

                model = BioMedCLIPLoRALPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "CLIP":
                from model.clip import CLIPLoRALPForDiagnosis

                model = CLIPLoRALPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BLIP2-2.7b":
                from model.blip2 import BLIP2LPLoRAForDiagnosis

                model = BLIP2LPLoRAForDiagnosis(args=args, text=text, num_classes=num_classes)
            else:
                raise NotImplementedError()
            
        elif args.usage in ["clip-zs", "clip-img-lora"]:
            if args.model == "BLIP":
                from model.blip import BLIPForDiagnosis

                model = BLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "CLIP":
                from model.clip import CLIPForDiagnosis

                model = CLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BioMedCLIP":
                from model.biomedclip import BiomedCLIPForDiagnosis

                model = BiomedCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "MedCLIP":
                from model.medclip import MedCLIPForDiagnosis

                model = MedCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PMCCLIP":
                model = PMCCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "BLIP2-2.7b":
                from model.blip2 import BLIP2ForDiagnosis

                model = BLIP2ForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PLIP":
                from model.plip import PLIPForDiagnosis

                model = PLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "SigLIP":
                from model.siglip import SiglipForDiagnosis

                model = SiglipForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "MedSigLIP":
                from model.medsiglip import MedSigLIPForDiagnosis

                model = MedSigLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            elif args.model == "PubMedCLIP":
                from model.pubmedclip import PubMedCLIPForDiagnosis

                model = PubMedCLIPForDiagnosis(args=args, text=text, num_classes=num_classes)
            else:
                raise NotImplementedError()
        elif args.usage in ["clip-adapter"]:
            raise NotImplementedError()
            text = get_prototype(args)
            text = ["a photo of {}".format(txt) for txt in text]
            if args.model == "CLIP":
                # CLIPAdapterWrapper needs the base CLIP model instance
                from model.clip_adapter import CLIPAdapterWrapper

                clip_model_instance = CLIPForDiagnosis(text=text, num_classes=num_classes)
                model = CLIPAdapterWrapper(text=text, num_classes=num_classes, clip_model=clip_model_instance.model)
            else:
                raise NotImplementedError(f"CLIP-Adapter not implemented for model {args.model}")
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return model
