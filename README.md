# MedVLMBench: A Unified Benchmark for Medical Vision-Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2506.17337"><img src="https://img.shields.io/badge/arXiv-2506.17337-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/ubc-tea/MedVLMBench/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.8%2B-green.svg" alt="Python">
  <img src="https://img.shields.io/github/stars/ubc-tea/MedVLMBench?style=social" alt="Stars">
  <a href="https://github.com/FairMedFM/FairMedFM"><img src="https://img.shields.io/badge/Companion-FairMedFM-orange.svg" alt="FairMedFM"></a>
</p>

**MedVLMBench** is the first unified benchmark for systematically evaluating generalist and medical-specialist Vision-Language Models (VLMs). It covers **35+ models**, **14 datasets**, and **3 task types** (VQA, diagnosis, captioning) across radiology, pathology, dermatology, and ophthalmology — with support for off-the-shelf inference, linear probing, LoRA fine-tuning, and multi-agent reasoning.

---

## Highlights

- **35+ models supported** — CLIP-based (BioMedCLIP, MedCLIP, PLIP, SigLIP …) and generative (LLaVA, MedGemma, Qwen3-VL, InternVL3.5, GLM-4.5V, Molmo, Llama-4, Gemini 3.1 Pro, GPT-5.2, HuatuoGPT-Vision, MAIRA-2 …)
- **14 medical datasets** — SLAKE, PathVQA, VQA-RAD, MedXpertQA, OmniMedVQA, PneumoniaMNIST, HAM10000, CheXpert, MIMIC-CXR and more
- **3 evaluation tasks** — Visual Question Answering (VQA), Diagnostic Classification, Report Captioning
- **Flexible fine-tuning** — off-the-shelf (OTS), linear probing (LP), LoRA, and full fine-tuning
- **Multi-agent reasoning** — MDAgent and UCAgent wrappers for chain-of-thought and debate-style inference
- **Reproducible results** — full CLI + Jupyter notebook tutorials included

## News

- **2026-07** Added 12 new generalist and medical-specialist VLMs: Qwen3-VL, InternVL3.5, GLM-4.1V-Thinking, GLM-4.5V, Molmo, Llama-4, Gemini 3.1 Pro, GPT-5.2, HuatuoGPT-Vision (Qwen2.5-VL revision), MedVLM-R1, CheXagent-2, and MAIRA-2
- **2025-06** Paper released on arXiv ([2506.17337](https://arxiv.org/abs/2506.17337))
- **2025-06** Added MedXpertQA and OmniMedVQA benchmark datasets
- **2025-06** Added MDAgent (multi-specialist reasoning) and UCAgent (hierarchical debate) wrappers
- **2025-06** Added InternVL3, Gemma3, Qwen2-VL, Qwen2.5-VL, Lingshu, o3, Gemini 2.5 Pro

---

## Companion Benchmark: FairMedFM

> **Evaluating fairness of medical FMs?** See our companion benchmark [**FairMedFM**](https://github.com/FairMedFM/FairMedFM) — the first fairness benchmark covering 20 medical imaging FMs across 17 datasets with bias metrics over sex, race, and age.

MedVLMBench and FairMedFM form a **two-part evaluation suite** for medical foundation models — capability and fairness, measured on the same models and datasets.

| | [MedVLMBench](https://github.com/ubc-tea/MedVLMBench) | FairMedFM |
|---|---|---|
| **Focus** | Capability: accuracy, AUROC, VQA scores | Fairness across sex, race, age |
| **Model paradigm** | Generative VLMs + discriminative models | Discriminative FMs (CLIP, SAM variants) |
| **Tasks** | VQA, Diagnosis, Captioning | Classification, Segmentation |
| **Scale** | 30+ VLMs · 14 datasets | 20 FMs · 17 datasets |

**Models evaluated in both**: BioMedCLIP · MedCLIP · PLIP · SigLIP · MedSigLIP · CLIP · BLIP · BLIP2 · PubMedCLIP

**Datasets in both**: HAM10000 · CheXpert · MIMIC-CXR · FairVLMed10k · GF3300 · PAPILA

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Download Datasets and Models](#download-datasets-and-models)
- [Available Models and Datasets](#available-models-and-datasets)
  - [Datasets](#datasets)
  - [Models](#models)
- [Usage](#usage)
  - [Notebook Tutorials](#notebook-tutorials)
  - [Command-Line Interface](#command-line-interface)
- [Abstract](#abstract)
- [Citation](#citation)

---

## Getting Started

### Prerequisites

Python 3.11, PyTorch 2.1+, CUDA 11.8+.

### Installation

**Option A — pip**

```bash
git clone https://github.com/ubc-tea/MedVLMBench.git
cd MedVLMBench
pip install -r requirements.txt
```

**Option B — conda**

```bash
git clone https://github.com/ubc-tea/MedVLMBench.git
cd MedVLMBench
conda env create -f environment.yml
conda activate medvlmbench
```

> **Optional fast-attention deps** (require CUDA build tools): uncomment `flash-attn` and/or `xformers` in `requirements.txt` before installing.

### Download Datasets and Models

All pretrained models should be stored under `MedVLMBench/pretrained_models`, and all data under `MedVLMBench/data`.

```bash
mkdir pretrained_models data
```

**Example: LLaVA-1.5**

```bash
cd pretrained_models
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
cd ..
```

**Example: MedXpertQA**

```bash
cd data
git clone https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA
```

**Example: OmniMedVQA**

```bash
cd data
git clone https://huggingface.co/datasets/foreverbeliever/OmniMedVQA
```

---

## Available Models and Datasets

### Datasets

| Dataset | Task | Modality |
|---|---|---|
| SLAKE | VQA | Radiology |
| PathVQA | VQA | Pathology |
| VQA-RAD | VQA | Radiology |
| FairVLMed10k | VQA / Diagnosis / Captioning | Ophthalmology |
| MedXpertQA | VQA (multi-choice) | Multi-modal |
| OmniMedVQA | VQA (multi-choice) | Multi-modal |
| MIMIC-CXR | Captioning | Radiology |
| PneumoniaMNIST | Diagnosis | Radiology |
| BreastMNIST | Diagnosis | Radiology |
| DermaMNIST | Diagnosis | Dermatology |
| Camelyon17 | Diagnosis | Pathology |
| HAM10000 | Diagnosis | Dermatology |
| CheXpert | Diagnosis | Radiology |
| ChestXray14 | Diagnosis | Radiology |
| GF3300 | Diagnosis | Ophthalmology |
| PAPILA | Diagnosis | Ophthalmology |
| Drishti | Diagnosis | Ophthalmology |

### Models

<details>
<summary><b>Generative VLMs (VQA / Captioning)</b></summary>

| Model | Evaluation | Training |
|---|---|---|
| o3 (OpenAI) | Done | NA |
| GPT-5.2 (OpenAI) | Added — untested* | NA |
| Gemini 2.5 Pro | Done | NA |
| Gemini 3.1 Pro | Added — untested* | NA |
| InternVL3 | Done | Coming Soon |
| InternVL3.5 | Added — untested* | Coming Soon |
| LLaVA-1.5 | Done | Done |
| LLaVA-Med | Done | Done |
| Gemma3 | Done | Coming Soon |
| MedGemma | Done | Done |
| Qwen2-VL | Done | Coming Soon |
| Qwen2.5-VL | Done | Coming Soon |
| Qwen3-VL | Added — untested* | Coming Soon |
| GLM-4.1V-Thinking | Added — untested* | Coming Soon |
| GLM-4.5V | Added — untested* | Coming Soon |
| Molmo | Added — untested* | Coming Soon |
| Llama-4 | Added — untested* | Coming Soon |
| NVILA | Done | Done |
| VILA-M3 | Done | Done |
| VILA1.5 | Done | Done |
| Lingshu | Done | Done |
| HuatuoGPT-Vision | Added — untested* | Coming Soon |
| MedVLM-R1 | Added — untested* | Coming Soon |
| CheXagent-2 | Added — untested* | Coming Soon |
| MAIRA-2 | Added — untested* | Coming Soon |
| XrayGPT | Done | Done |
| BLIP | Done | Done |
| BLIP2-2.7b | Done | Done |

*Added — untested: wrapper code is in place and import/syntax-checked, but not yet run end-to-end against downloaded weights. Please report issues.

</details>

<details>
<summary><b>Contrastive / CLIP-based Models (Diagnosis)</b></summary>

| Model | Evaluation | Training |
|---|---|---|
| BioMedCLIP | Done | Done |
| CLIP | Done | Done |
| MedCLIP | Done | Done |
| PMCCLIP | Done | Done |
| PLIP | Done | Done |
| MedSigLIP | Done | Done |
| PubMedCLIP | Done | Done |
| SigLIP | Done | Done |

</details>

> **License note:** `Llama-4` (Meta license) and `MAIRA-2` (Microsoft Research License Agreement) require accepting a gated license on Hugging Face before download. `CheXagent-2` and `MAIRA-2` are released under non-commercial terms (CC-BY-NC-4.0 / MSRLA) — research use only.

---

## Usage

`run_eval.py` is the main entry point for evaluation. `run_train.py` is the main entry point for fine-tuning.

### Notebook Tutorials

| Feature | Notebook |
|---|---|
| Off-the-shelf Diagnosis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ubc-tea/MedVLMBench/blob/main/examples/MedVLMBench_OTS_Diagnosis.ipynb) |
| Off-the-shelf VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ubc-tea/MedVLMBench/blob/main/examples/MedVLMBench_OTS_VQA.ipynb) |
| LP Diagnosis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ubc-tea/MedVLMBench/blob/main/examples/MedVLMBench_LP_Diagnosis.ipynb) |
| LoRA Adaptation VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ubc-tea/MedVLMBench/blob/main/examples/MedVLMBench_LoRA_VQA.ipynb) |

### Command-Line Interface

#### Off-the-shelf Evaluation

<details>
<summary><b>Diagnosis (zero-shot CLIP)</b></summary>

```bash
python run_eval.py \
  --task diagnosis --usage clip-zs --dataset PAPILA --split test \
  --image_path ./data \
  --exp_path ./log \
  --model CLIP --model_path "original_pretrained" \
  --save_pred \
  --cache_dir ./cache
```

</details>

<details>
<summary><b>VQA (generative model)</b></summary>

```bash
python run_eval.py \
  --task vqa --dataset SLAKE --split test \
  --image_path ./data/SLAKE/imgs \
  --model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
  --exp_path ./log \
  --cache_dir ./cache \
  --save_pred
```

</details>

<details>
<summary><b>MDAgent multi-specialist reasoning</b></summary>

Wrap any supported VLM backbone with multi-agent reasoning by adding `--usage mdagent`:

```bash
python run_eval.py \
  --task vqa --dataset VQA-RAD --split test \
  --image_path ./data \
  --model Qwen2-VL \
  --model_path ./pretrained_models/Qwen2-VL-2B-Instruct \
  --usage mdagent \
  --mdagent_mode adaptive \
  --exp_path ./log \
  --cache_dir ./cache \
  --save_pred
```

MDAgent modes: `basic`, `intermediate`, `advanced`, `adaptive` (recommended). When `--save_pred` is set, the output file includes the full reasoning trace per sample.

</details>

<details>
<summary><b>UCAgent hierarchical debate reasoning</b></summary>

```bash
python run_eval.py \
  --task vqa --dataset MedXpertQA --split test \
  --image_path ./data/MedXpertQA \
  --model MedGemma \
  --model_path ./pretrained_models/medgemma-4b-it \
  --usage ucagent \
  --exp_path ./log \
  --cache_dir ./cache \
  --save_pred
```

UCAgent runs a 3-level hierarchical diagnosis: two independent expert assessments → senior expert verification → critic-panel debate with leader adjudication.

</details>

#### Fine-tuning

<details>
<summary><b>Linear probing (diagnosis)</b></summary>

```bash
python run_train.py \
  --task diagnosis --usage lp --dataset HAM10000 --split train \
  --image_path ./data \
  --output_dir ./log \
  --model CLIP --model_path not_given \
  --cache_dir ./cache \
  --num_train_epochs 50 \
  --learning_rate 5e-5
```

Other fine-tuning modes: `img-lora-lp` (LP + image encoder LoRA), `clip-img-lora` (CLIP image encoder LoRA).

</details>

<details>
<summary><b>LoRA fine-tuning (VQA)</b></summary>

```bash
deepspeed run_train.py \
  --peft lora --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
  --deepspeed ./script/zero3.json \
  --task vqa --dataset SLAKE \
  --model LLaVA-1.5 --version v1 \
  --image_path ./data/SLAKE/imgs \
  --model_path ./pretrained_models/llava-v1.5-7b \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./log \
  --cache_dir ./cache \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --tune_modules L
```

</details>

---

## Abstract

>**Background:** Vision–Language Models (VLMs) have shown promise in automating image diagnosis and interpretation in clinical settings. However, developing medical-specialist VLMs requires substantial computational resources and carefully curated datasets, and it remains unclear under which conditions generalist and medical specialist VLMs each perform best.

>**Methods:** This paper introduces MedVLMBench, the first unified benchmark for systematically evaluating generalist and medical-specialist VLMs. We assessed 18 models spanning contrastive and generative paradigms on 10 publicly available datasets across radiology, pathology, dermatology, and ophthalmology, encompassing 144 diagnostic and 80 VQA settings. MedVLMBench focuses on assessing both in-domain (ID) and out-of-domain (OOD) performance, with off-the-shelf and parameter-efficient fine-tuning (e.g., linear probing, LoRA). Diagnostic classification tasks were evaluated using AUROC, while VQA tasks were assessed with BLEU-1, ROUGE-L, Exact Match, F1 Score, and GPT-based semantic scoring.

>**Results:** Off-the-shelf medical VLMs generally outperformed generalist VLMs on in-domain tasks. However, with lightweight fine-tuning, general-purpose VLMs achieved superior performance in most in-domain evaluations and demonstrated better generalization on out-of-domain tasks. Fine-tuning required only **3% of the parameters** associated with full medical pretraining.

>**Conclusions:** Efficiently fine-tuned generalist VLMs can match or surpass medical-specialist VLMs in most tasks, offering a scalable and cost-effective pathway for clinical AI development.

---

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@article{zhong2025can,
  title={Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights},
  author={Zhong, Yuan and Jin, Ruinan and Li, Xiaoxiao and Dou, Qi},
  journal={arXiv preprint arXiv:2506.17337},
  year={2025}
}
```

If you also use our companion benchmark **FairMedFM** for fairness evaluation, please cite:

```bibtex
@article{jin2024fairmedfm,
  title={FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models},
  author={Jin, Ruinan and Xu, Zikang and Zhong, Yuan and Yao, Qiongsong and Dou, Qi and Zhou, S Kevin and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2407.00983},
  year={2024}
}
```
