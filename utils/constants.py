# tasks
TASKS = ["vqa", "caption", "diagnosis"]


# models
CLIP_MODELS = [
    "BLIP",
    "BLIP2-2.7b",
    "BioMedCLIP",
    "CLIP",
    "MedCLIP",
    "PMCCLIP",
    "PLIP",
    "MedSigLIP",
    "PubMedCLIP",
    "SigLIP",
    "SigLIP2",
    "DermLIP",
    "EyeCLIP",
    "CONCH",
]

VISION_FOUNDATION_MODELS = [
    "DINOv2", "DINOv3", "RAD-DINO", "AIMv2", "UNI2", "Virchow2",
    "Prov-GigaPath", "RETFound",
]

LANGUAGE_MODELS = [
    "LLaVA-1.5",
    "LLaVA-Med",
    "Quilt-LLaVA",
    "Gemma3",
    "MedGemma",
    "Qwen2-VL",
    "Qwen25-VL",
    "Patho-R1",
    "InternVL3",
    "InternVL3.5",
    "Qwen3-VL",
    "GLM-4.1V-Thinking",
    "GLM-4.5V",
    "Molmo",
    "Llama-4",
    "HuatuoGPT-Vision",
    "MedVLM-R1",
    "CheXagent-2",
    "MAIRA-2",
    "XGenMiniV1",
    "XrayGPT",
    "NVILA",
    "VILA-M3",
    "VILA1.5",
    "o3",
    "gemini-2.5-pro",
    "gemini-3.1-pro",
    "gpt-5.2",
    "Lingshu",
]

MODELS = CLIP_MODELS + VISION_FOUNDATION_MODELS + LANGUAGE_MODELS


# datasets
VQA_DATASETS = ["SLAKE", "PathVQA", "VQA-RAD", "Harvard-FairVLMed10k", "MedXpertQA", "OmniMedVQA"]
CAPTION_DATASETS = ["HarvardFairVLMed10k", "MIMIC_CXR"]
DIAGNOSIS_DATASETS = [
    "PneumoniaMNIST",
    "BreastMNIST",
    "DermaMNIST",
    "Camelyon17",
    "Drishti",
    "HAM10000",
    "ChestXray",
    "GF3300",
    "CheXpert",
    "PAPILA",
    "HarvardFairVLMed10k",
]

DATASETS = VQA_DATASETS + CAPTION_DATASETS + DIAGNOSIS_DATASETS
