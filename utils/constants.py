# tasks
TASKS = ["vqa", "caption", "diagnosis"]


# models
CLIP_MODELS = ["BLIP", "BLIP2-2.7b", "BioMedCLIP", "CLIP", "MedCLIP", "PMCCLIP", "PLIP", "MedSigLIP", "PubMedCLIP", "SigLIP"]

LANGUAGE_MODELS = [
    "LLaVA-1.5",
    "LLaVA-Med",
    "Gemma3",
    "MedGemma",
    "Qwen2-VL",
    "Qwen25-VL",
    "InternVL3",
    "XGenMiniV1",
    "XrayGPT",
    "NVILA",
    "VILA-M3",
    "VILA1.5",
    "o3",
    "gemini-2.5-pro",
    "Lingshu",
]

MODELS = CLIP_MODELS + LANGUAGE_MODELS


# datasets
VQA_DATASETS = ["SLAKE", "PathVQA", "VQA-RAD", "Harvard-FairVLMed10k"]
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
