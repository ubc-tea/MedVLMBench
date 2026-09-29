import json
import os, sys
import random
import argparse
import traceback
from utils import constants

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils import basics
from model import get_model
from dataset import get_dataset

from train import get_train_engine
from utils.experiment_tracking import ExperimentTracker


@dataclass
class Arguments(transformers.TrainingArguments):
    # data
    task: str = field(default="vqa")
    dataset: str = field(default="SLAKE")
    peft: str = field(default=None)
    image_path: str = field(default="")
    image_aspect_ratio: str = field(default="pad")

    # train
    optim: str = field(default="adamw_torch")
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    tune_modules: str = "VML"  # V for vision tower, M for multimodal projector, L for LLM
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    num_train_epochs: int = None
    learning_rate: float = 3e-5
    train_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of the official training split to use (0 < fraction <= 1)."},
    )
    fraction_seed: int = field(
        default=42,
        metadata={"help": "Seed used only for deterministic training-set subsampling."},
    )
    curve_run: bool = field(
        default=False,
        metadata={"help": "Keep the 100 percent learning-curve run separate from earlier full-data runs."},
    )
    flops_profile_batches: int = field(
        default=1,
        metadata={"help": "Number of training batches to profile for FLOPs/sample; set 0 to disable."},
    )

    # evaluation
    eval_print_freq: int = 100
    save_pred: bool = False
    save_total_limit: int = 2

    mm_projector_lr: Optional[float] = None  # for LLMs
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    group_by_modality_length: bool = field(default=False)
    deepspeed_plugin: Optional[str] = field(default=None)

    # network
    model: str = field(default="LLaVA")
    version: str = field(default="v1")
    context_length: int = field(default=77)
    model_path: str = field(default=None, metadata={"help": "explicitly indentify checkpoint path to resume."})
    model_base: str = field(default=None)
    patho_r1_training_authorized: bool = field(default=False, metadata={"help": "Confirm written permission from Patho-R1 rights holders before fine-tuning its restricted checkpoint."})
    freeze_backbone: bool = field(default=False)
    usage: str = field(default=None)

    ## LlaVA
    tune_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")

    ## VILA
    longvila_sampler: bool = field(default=False)
    seq_parallel_size: int = field(
        default=-1,
        metadata={"help": "The degree of sequence parallelism (SP). SP is disabled by default (value: -1). "},
    )
    image_aspect_ratio: Optional[str] = "resize"
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    num_time_tokens: int = field(default=0)
    time_token_format: str = field(default="<t{t}>")
    soft_ce_std: float = field(default=1.0)
    max_num_images: int = field(default=6)
    debug_e2e: bool = field(
        default=False,
        metadata={"help": "Whether enter debug mode."},
    )

    # misc
    # exp_path: str = field(default="")
    # device: Optional[str] = field(default="cuda")
    cache_dir: Optional[str] = field(default=None)
    if_wandb: Optional[str] = False
    wandb_name: Optional[str] = field(default=None)
    split: Optional[str] = field(default="train")


def setup_args(args):
    assert args.task in constants.TASKS, f"Task {args.task} is not supported. Supported tasks: {constants.TASKS}"
    assert args.model in constants.MODELS, f"Model {args.model} is not supported. Supported models: {constants.MODELS}"
    assert (
        args.dataset in constants.DATASETS
    ), f"Dataset {args.task} is not supported. Supported datasets: {constants.DATASETS}"
    assert args.dataset in getattr(
        constants, f"{str.upper(args.task)}_DATASETS"
    ), f"dataset {args.dataset} is not supported for task {args.task}"

    save_folder_name = f"train_{args.peft}_{args.tune_modules}_seed{args.seed}"

    if not 0 < args.train_fraction <= 1:
        raise ValueError("train_fraction must satisfy 0 < train_fraction <= 1")

    if "LLaVA" in args.model and args.tune_modules == "M":
        args.peft = ""
        print(args.peft)

        save_folder_name = f"train_{args.peft}_{args.tune_modules}_seed{args.seed}"

    if "llava" in args.model.lower():
        assert (
            args.gradient_checkpointing is False
        ), "Currently there is a bug when training visual tower using peft + gradient checkpointing. For more info: https://github.com/huggingface/peft/issues/1402"

        # HF Trainer falls back to DataParallel for plain multi-GPU `python run_train.py`
        # launches. LLaVA produces variable sequence lengths across replicas after
        # multimodal token expansion, which breaks DataParallel gather. Keep the
        # plain launch on a single visible GPU unless the user explicitly uses a
        # distributed launcher such as torchrun/deepspeed.
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not args.deepspeed:
            args._n_gpu = 1

        if args.model == "LLaVA-1.5":
            save_folder_name += "_llava"
        if args.model == "LLaVA-Med":
            save_folder_name += "_llava_mistral"
        if args.model == "Quilt-LLaVA":
            save_folder_name += "_quilt_llava"
    elif args.model == "NVILA":
        save_folder_name = f"train_{args.peft}_{args.tune_modules}_seed{args.seed}_nvila"
    elif args.model == "VILA1.5":
        save_folder_name = f"train_{args.peft}_{args.tune_modules}_seed{args.seed}_vila"
    elif args.model == "VILA-M3":
        save_folder_name = f"train_{args.peft}_{args.tune_modules}_seed{args.seed}_vila_m3"
    elif args.model == "Lingshu":
        save_folder_name = f"train_{args.peft}_{args.tune_modules}_seed{args.seed}_lingshu"
    elif args.model == "Patho-R1":
        save_folder_name = f"train_{args.peft}_{args.tune_modules}_seed{args.seed}_patho_r1"
    elif args.task == "diagnosis":
        save_folder_name = f"train_{args.usage}_seed{args.seed}"

    if args.train_fraction < 1 or args.curve_run:
        fraction_label = f"{args.train_fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        save_folder_name += f"_frac{fraction_label}_fseed{args.fraction_seed}"

    args.output_dir = os.path.join(
        args.output_dir,
        args.task,
        args.dataset,
        args.model,
        save_folder_name,
    )
    args.split = "train"
    args.tune_mm_mlp_adapter = "M" in args.tune_modules

    if args.peft == "lora":
        args.lora_enable = True
    else:
        args.lora_enable = False

    basics.creat_folder(args.output_dir)

    return args


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]
    args = setup_args(args)

    logger = basics.setup_logger("train", args.output_dir, "train.log", screen=True, tofile=True)
    logger.info("Using following arguments for training.")
    logger.info(args)

    args.logger = logger

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_wrapped = get_model(args=args, device=args.device.type)
    model_wrapped.load_for_training(args.model_path)

    # Checking model param here may be unreliable. The param state may be changed when initializing HF trainers using training configs.
    # total_params = sum(p.numel() for p in model_wrapped.parameters())
    # trainable_params = sum(p.numel() for p in model_wrapped.parameters() if p.requires_grad)
    # trainable_percentage = 100 * trainable_params / total_params
    # args.logger.info(f"Total number of parameters: {total_params/1e6:.2f}M")
    # args.logger.info(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    # args.logger.info(f"Trainable parameters percentage: {trainable_percentage:.2f}%")

    dataset_image_processor = getattr(model_wrapped, "image_processor_callable", None) or getattr(model_wrapped, "image_processor", None)
    dataset = get_dataset(args, image_processor_callable=dataset_image_processor)
    tracker = ExperimentTracker(
        output_dir=args.output_dir,
        phase="train",
        args=args,
        model=model_wrapped,
        dataset=dataset,
    )
    tracker.start()
    try:
        train_engine = get_train_engine(args, model_wrapped=model_wrapped, dataset=dataset)
        tracker.attach_trainer(train_engine.hf_trainer)
        if args.task == "diagnosis" and hasattr(train_engine.hf_trainer, "eval_engine"):
            train_engine.hf_trainer.eval_engine.flop_tracker = tracker
        train_engine.train()
    except BaseException as exc:
        tracker.finish(status="failed", error="".join(traceback.format_exception_only(type(exc), exc)).strip())
        raise
    else:
        tracker.finish(status="completed")

    logger.info("End of the training")
