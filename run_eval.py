import json
import os
import random
import argparse
from utils import constants

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import basics
from model import get_model
from dataset import get_dataset
from eval import get_eval_engine


def collect_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--task", default="vqa", choices=constants.TASKS)
    parser.add_argument(
        "--dataset",
        default="SLAKE",
        choices=constants.DATASETS,
    )
    parser.add_argument("--image_path", type=str, default="", help="local path to images")
    parser.add_argument("--split", type=str, default="all", help="dataset split for evaluation")

    # evaluation
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=10, help="logging frequency during evaluation")
    parser.add_argument("--save_pred", action="store_true", help="whether to save predictions during evaluation")
    parser.add_argument("--gpt_eval", action="store_true", help="whether to use GPT for evaluation")

    parser.add_argument("--hash_id", type=str, default="")

    # network
    parser.add_argument(
        "--model",
        default="BLIP",
        choices=constants.MODELS,
    )
    parser.add_argument("--context_length", default=77)
    parser.add_argument("--model_path", type=str, default=None, help="explicitly indentify checkpoint path to resume.")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--usage", type=str, default=None)

    # misc
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--eval_print_freq", type=int, default=100, help="logging frequency (step)")
    parser.add_argument("--exp_path", type=str, default="./output")
    parser.add_argument("--wandb_name", type=str, default="baseline")
    parser.add_argument("--if_wandb", type=bool, default=False)

    args = parser.parse_args()

    assert args.dataset in getattr(
        constants, f"{str.upper(args.task)}_DATASETS"
    ), f"dataset {args.dataset} is not supported for task {args.task}"

    args.output_dir = os.path.join(
        args.exp_path, args.task, args.dataset, args.model, f"eval_seed{args.seed}", os.path.basename(args.model_path)
    )

    basics.creat_folder(args.output_dir)

    if args.cache_dir is not None:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

    return args


if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 128

    args = collect_args()

    logger = basics.setup_logger("eval", args.output_dir, "eval.log", screen=True, tofile=True)
    logger.info("Using following arguments for evaluation.")
    logger.info(args)

    args.logger = logger

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_wrapped = get_model(args=args, device=args.device)

    total_params = sum(p.numel() for p in model_wrapped.parameters())
    args.logger.info(f"Total number of parameters: {total_params/1e6:.2f}M")

    if args.model_path != "original_pretrained":
        model_wrapped.load_from_pretrained(model_path=args.model_path, device=args.device)

    dataset = get_dataset(args, getattr(model_wrapped, "image_processor", None))

    eval_engine = get_eval_engine(args=args, dataset=dataset)
    eval_engine.evaluate(args=args, model=model_wrapped)

    args.logger.info("End of the evaluation")
