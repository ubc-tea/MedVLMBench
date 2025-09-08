# MedVLMBench: A Unified Benchmark for Generalist and Specialist Medical Vision-Language Models

MedVLMBench is the first unified benchmark for systematically evaluating generalist and medical-specialist Vision-Language Models (VLMs). This repository provides the code and resources to reproduce the experiments and extend the benchmark.

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
- [License](#license)


## Abstract

> **Background:** Vision–Language Models (VLMs) have shown promise in automating image diagnosis and interpretation in clinical settings. However, developing medical-specialist VLMs requires substantial computational resources and carefully curated datasets, and it remains unclear under which conditions generalist and medical specialist VLMs each perform best.
>
> **Methods:** This paper introduces MedVLMBench, the first unified benchmark for systematically evaluating generalist and medical-specialist VLMs. We assessed 18 models spanning contrastive and generative paradigms on 10 publicly available datasets across radiology, pathology, dermatology, and ophthalmology, encompassing 144 diagnostic and 80 VQA settings. MedVLMBench focusing on assessing both in-domain (ID) and out-of-domain (OOD) performance, with off-the-shelf and parameter-efficient fine-tuning (e.g., linear probing, LoRA). Diagnostic classification tasks were evaluated using AUROC, while visual question answering (VQA) tasks were assessed with BLEU-1, ROUGE-L, Exact Match, F1 Score, and GPT-based semantic scoring, covering both open- and closed-ended formats. Computational efficiency was estimated relative to the cost of full medical pretraining.
>
> **Results:** As expected, off-the-shelf medical VLMs generally outperformed generalist VLMs on ID tasks given their pretraining. However, with lightweight fine-tuning, general-purpose VLMs achieved superior performance in most of ID task evaluations and demonstrated better generalization on OOD tasks in approximately all comparisons. Fine-tuning required only 3% of the total parameters associated with full medical pretraining. In contrast, fine-tuned medical VLMs showed degraded performance even on ID tasks when subjected to rigorous hyperparameter optimization, further highlighting their limited adaptability.
>
> **Conclusions:** This study highlights the complementary strengths of medical-specialist and generalist VLMs. Specialists remain valuable in modality-aligned use cases, but we find that efficiently fine-tuned generalist VLMs can achieve comparable or even superior performance in most tasks, particularly when transferring to unseen or rare OOD medical modalities. These results suggest that generalist VLMs, rather than being constrained by their lack of medical-specific pretraining, may offer a scalable and cost-effective pathway for advancing clinical AI development.


## Getting Started

### Prerequisites

Ensure you have an environment with Python and the necessary dependencies. You can set up a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate medvlmbenc
```

### Installation

Clone the repository:

```bash
git clone https://github.com/Nanboy-Ronan/MedVLMBench.git
cd MedVLMBench
```

### Download Datasets and Models

All pretrained models should be stored under `MedVLMBench/pretrained_models`, and all data should be stored under `MedVLMBench/data`.

```bash
mkdir pretrained_models
mkdir data
```

**Example: Downloading LLaVA**

```bash
cd pretrained_models
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b
cd ..
```

## Available Models and Datasets
This code base mainly supports the image diagnostics and the VQA tasks. It also supports image captioning, which does not report the results in the paper.

<details>
<summary><b>Supported Datasets</b></summary>

| Dataset | Type | Status |
|---|---|---|
| SLAKE | VQA | Done |
| PathVQA | VQA | Done |
| VQA-RAD | VQA | Done |
| FairVLMed | VQA | Done |
| PneumoniaMNIST | Diagnosis | Done |
| BreastMNIST | Diagnosis | Done |
| DermaMNIST | Diagnosis | Done |
| Camelyon17 | Diagnosis | Done |
| HAM10000 | Diagnosis | Done |
| Drishti | Diagnosis | Done |
| ChestXray | Diagnosis | Done |
| GF3300 | Diagnosis | Done |
| CheXpert | Diagnosis | Done |
| PAPILA | Diagnosis | Done |
| FairVLMed | Diagnosis | Done |

</details>

<details>
<summary><b>Supported Models</b></summary>

| Model | Type | Evaluation | Training |
|---|---|---|---|
| o3 | VQA | Done | NA |
| Gemini 2.5 Pro | VQA | Done | NA |
| InternVL3 | VQA | Done | Coming Soon |
| LLaVA-1.5 | VQA | Done | Done |
| LLaVA-Med | VQA | Done | Done |
| Gemma3 | VQA | Done | Coming Soon |
| MedGemma | VQA | Done | Done |
| Qwen2-VL | VQA | Done | Coming Soon |
| Qwen25-VL | VQA | Done | Coming Soon |
| NVILA | VQA | Done | Done |
| VILA-M3 | VQA | Done | Done |
| VILA1.5 | VQA | Done | Done |
| Lingshu | VQA | Done | Done |
| BLIP | Diagnosis/VQA | Done | Done |
| BLIP2 | Diagnosis/VQA | Done | Done |
| XrayGPTVQA | Diagnosis/VQA | Done | Done |
| BioMedCLIP | Diagnosis | Done | Done |
| CLIP | Diagnosis | Done | Done |
| MedCLIP | Diagnosis | Done | Done |
| PMCCLIP | Diagnosis | Done | Done |
| PLIP | Diagnosis | Done | Done |
| MedSigLIP | Diagnosis | Done | Done |
| PubMedCLIP | Diagnosis | Done | Done |
| SigLIP | Diagnosis | Done | Done |

</details>

## Usage

`run_train.py` is the major entry for training all models (including the lightweight adaptation).
`run_eval.py` is the major entry for off the shelf evaluation of all models.

### Notebook Tutorials

We offer some examples of how to use our package through the notebook.

| Feature | Notebook |
|---|---|
| Off-the-shelf Diagnosis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/Nanboy-Ronan/MedVLMBench/blob/main/examples/MedVLMBench_OTS_Diagnosis.ipynb) |
| Off-the-shelf VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/Nanboy-Ronan/MedVLMBench/blob/main/examples/MedVLMBench_OTS_VQA.ipynb) |
| LP Diagnosis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/Nanboy-Ronan/MedVLMBench/blob/main/examples/MedVLMBench_LP_Diagnosis.ipynb) |
| LoRA Adaptation VQA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/Nanboy-Ronan/MedVLMBench/blob/main/examples/MedVLMBench_LoRA_VQA.ipynb) |

### Command-Line Interface

This section provides detailed documentation for the command-line arguments used in `run_eval.py` and `run_train.py`.

#### `run_eval.py`

This script is used for evaluating the performance of a trained model on a given dataset.

**Usage:**

```bash
python run_eval.py [OPTIONS]
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--task` | str | `vqa` | The task to perform. Choices: `vqa`, `diagnosis`, `caption`. |
| `--dataset` | str | `SLAKE` | The dataset to use for evaluation. |
| `--image_path` | str | | The local path to the directory containing the images. |
| `--split` | str | `all` | The dataset split to use for evaluation (e.g., `test`, `val`). |
| `--seed` | int | `0` | The random seed for reproducibility. |
| `--print_freq` | int | `10` | The frequency of logging during evaluation. |
| `--save_pred` | bool | `False` | Whether to save the model's predictions. |
| `--gpt_eval` | bool | `False` | Whether to use GPT for evaluation (for VQA tasks). |
| `--hash_id` | str | | A unique hash ID for the experiment. |
| `--model` | str | `BLIP` | The model to evaluate. |
| `--context_length` | int | `77` | The context length for the model. |
| `--model_path` | str | | The path to the pretrained model checkpoint. |
| `--model_base` | str | | The base model name. |
| `--usage` | str | | The usage mode for the model. |
| `--device` | str | `cuda` | The device to use for evaluation (e.g., `cuda`, `cpu`). |
| `--cache_dir` | str | | The directory to cache pretrained models and other data. |
| `--eval_print_freq` | int | `100` | The logging frequency (in steps) during evaluation. |
| `--exp_path` | str | `./output` | The path to the experiment output directory. |
| `--wandb_name` | str | `baseline` | The name for the Weights & Biases run. |
| `--if_wandb` | bool | `False` | Whether to use Weights & Biases for logging. |

**Example:**

```bash
python run_eval.py \
--task vqa --dataset SLAKE --split test \
--image_path ./data/SLAKE/imgs \
--model LLaVA-1.5 --model_path ./pretrained_models/llava-v1.5-7b \
--exp_path ./log \
--cache_dir ./cache \
--save_pred
```

---

#### `run_train.py`

This script is used for training or fine-tuning a model on a given dataset.

**Usage:**

```bash
python run_train.py [OPTIONS]
```

**Arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `task` | str | `vqa` | The task to perform. Choices: `vqa`, `diagnosis`, `caption`. |
| `dataset` | str | `SLAKE` | The dataset to use for training. |
| `peft` | str | | The parameter-efficient fine-tuning method to use. |
| `image_path` | str | | The local path to the directory containing the images. |
| `image_aspect_ratio` | str | `pad` | The aspect ratio handling for images. |
| `optim` | str | `adamw_torch` | The optimizer to use for training. |
| `bits` | int | `16` | The number of bits to use for quantization. |
| `tune_modules` | str | `VML` | The modules to tune during training (V: vision, M: multimodal, L: language). |
| `lora_enable` | bool | `False` | Whether to enable LoRA for fine-tuning. |
| `lora_r` | int | `128` | The rank for LoRA. |
| `lora_alpha` | int | `256` | The alpha parameter for LoRA. |
| `lora_dropout` | float | `0.05` | The dropout rate for LoRA. |
| `lora_weight_path` | str | | The path to LoRA weights. |
| `lora_bias` | str | `none` | The bias to use for LoRA. |
| `num_train_epochs` | int | | The number of training epochs. |
| `learning_rate` | float | `3e-5` | The learning rate for training. |
| `eval_print_freq` | int | `100` | The frequency of logging during evaluation. |
| `save_pred` | bool | `False` | Whether to save the model's predictions. |
| `save_total_limit` | int | `2` | The maximum number of checkpoints to save. |
| `mm_projector_lr` | float | | The learning rate for the multimodal projector. |
| `remove_unused_columns` | bool | `False` | Whether to remove unused columns from the dataset. |
| `mpt_attn_impl` | str | `triton` | The implementation for MPT attention. |
| `model_max_length` | int | `2048` | The maximum sequence length for the model. |
| `double_quant` | bool | `True` | Whether to use double quantization. |
| `quant_type` | str | `nf4` | The quantization data type to use. |
| `group_by_modality_length` | bool | `False` | Whether to group inputs by modality length. |
| `deepspeed_plugin` | str | | The DeepSpeed plugin to use. |
| `model` | str | `LLaVA` | The model to train. |
| `version` | str | `v1` | The model version. |
| `context_length` | int | `77` | The context length for the model. |
| `model_path` | str | | The path to the pretrained model checkpoint. |
| `model_base` | str | | The base model name. |
| `freeze_backbone` | bool | `False` | Whether to freeze the backbone of the model. |
| `usage` | str | | The usage mode for the model. |
| `tune_mm_mlp_adapter` | bool | `False` | Whether to tune the multimodal MLP adapter. |
| `freeze_mm_mlp_adapter` | bool | `False` | Whether to freeze the multimodal MLP adapter. |
| `mm_vision_select_layer` | int | `-2` | The layer to select from the vision encoder. |
| `pretrain_mm_mlp_adapter` | str | | The path to a pretrained multimodal MLP adapter. |
| `mm_projector_type` | str | `mlp2x_gelu` | The type of multimodal projector to use. |
| `mm_use_im_start_end` | bool | `False` | Whether to use start and end tokens for images. |
| `mm_use_im_patch_token` | bool | `True` | Whether to use patch tokens for images. |
| `mm_patch_merge_type` | str | `flat` | The patch merging type to use. |
| `mm_vision_select_feature` | str | `patch` | The feature to select from the vision encoder. |
| `longvila_sampler` | bool | `False` | Whether to use the LongVILA sampler. |
| `seq_parallel_size` | int | `-1` | The sequence parallel size. |
| `vision_tower_lr` | float | | The learning rate for the vision tower. |
| `num_time_tokens` | int | `0` | The number of time tokens to use. |
| `time_token_format` | str | `<t{t}>` | The format for time tokens. |
| `soft_ce_std` | float | `1.0` | The standard deviation for soft cross-entropy. |
| `max_num_images` | int | `6` | The maximum number of images to use. |
| `debug_e2e` | bool | `False` | Whether to enable end-to-end debugging. |
| `cache_dir` | str | | The directory to cache pretrained models and other data. |
| `if_wandb` | bool | `False` | Whether to use Weights &amp; Biases for logging. |
| `wandb_name` | str | | The name for the Weights &amp; Biases run. |
| `split` | str | `train` | The dataset split to use for training. |

**Example:**

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
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 2048 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--tune_modules L
```


## Citation

If you find this repository useful, please consider citing our paper:

```
@article{zhong2025can,
  title={Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights},
  author={Zhong, Yuan and Jin, Ruinan and Li, Xiaoxiao and Dou, Qi},
  journal={arXiv preprint arXiv:2506.17337},
  year={2025}
}
```

## License 

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.