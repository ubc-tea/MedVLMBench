import json
import os
import random
import argparse
import warnings
from collections import defaultdict
from utils import constants

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

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
    parser.add_argument(
        "--mdagent_mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "basic", "intermediate", "advanced"],
        help="Reasoning mode used when --usage mdagent is enabled.",
    )

    # misc
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs to use for data-parallel evaluation")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--eval_print_freq", type=int, default=100, help="logging frequency (step)")
    parser.add_argument("--max_samples", type=int, default=None, help="optionally limit evaluation to the first N samples")
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=None,
        help="override generation length for VLM decoding",
    )
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

    if args.usage is not None:
        args.output_dir = os.path.join(args.output_dir, args.usage)

    basics.creat_folder(args.output_dir)

    if args.cache_dir is not None:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

    return args


def _worker_output_dir(base_output_dir, rank):
    return os.path.join(base_output_dir, f".worker_{rank}")


def _worker_payload_path(base_output_dir, rank):
    return os.path.join(base_output_dir, f".worker_{rank}.json")


def _eval_worker(rank, world_size, args_dict):
    args = argparse.Namespace(**args_dict)
    args.device = f"cuda:{rank}"
    args.output_dir = _worker_output_dir(args.output_dir, rank)
    basics.creat_folder(args.output_dir)

    logger = basics.setup_logger(
        f"eval_worker_{rank}",
        args.output_dir,
        "eval.log",
        screen=(rank == 0),
        tofile=True,
    )
    args.logger = logger

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.cuda.set_device(rank)

    model_wrapped = get_model(args=args, device=args.device)
    if args.model_path != "original_pretrained":
        model_wrapped.load_from_pretrained(model_path=args.model_path, device=args.device)

    dataset_image_processor = getattr(
        model_wrapped, "image_processor_callable", getattr(model_wrapped, "image_processor", None)
    )
    dataset = get_dataset(args, dataset_image_processor)
    shard_indices = list(range(rank, len(dataset), world_size))
    args.eval_header = f"GPU {rank} [{len(shard_indices)} samples]:"
    logger.info(f"Worker {rank} processing {len(shard_indices)} / {len(dataset)} samples on {args.device}.")
    eval_engine = get_eval_engine(args=args, dataset=dataset)
    eval_engine.evaluate(args=args, model=model_wrapped, indices=shard_indices, save_outputs=False)

    payload = eval_engine.export_state(model_wrapped)
    payload["num_samples"] = len(dataset)
    payload["shard_rank"] = rank
    payload["world_size"] = world_size

    with open(_worker_payload_path(os.path.dirname(args.output_dir), rank), "w") as fp:
        json.dump(payload, fp, indent=2)


def _merge_worker_payloads(args):
    payloads = []
    for rank in range(args.num_gpus):
        payload_path = _worker_payload_path(args.output_dir, rank)
        with open(payload_path, "r") as fp:
            payloads.append(json.load(fp))

    meter_totals = defaultdict(lambda: {"total": 0.0, "count": 0})
    records = []
    for payload in payloads:
        records.extend(payload.get("records", []))
        for name, meter in payload.get("meters", {}).items():
            meter_totals[name]["total"] += meter["total"]
            meter_totals[name]["count"] += meter["count"]

    results = {name: meter["total"] / meter["count"] for name, meter in meter_totals.items() if meter["count"] > 0}

    info_payload = payloads[0]
    info = {
        "model": [info_payload["model_name"]],
        "model_name": [os.path.basename(args.model_path)],
        "task": [args.task],
        "dataset": [info_payload["dataset_name"]],
        "model_type": [info_payload["model_type"]],
        "modality": [info_payload["dataset_modality"]],
        "size": [info_payload["num_samples"]],
    }
    info.update({k: [v] for k, v in results.items()})

    import pandas as pd

    pd.DataFrame(info).to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    if args.save_pred:
        with open(os.path.join(args.output_dir, "predictions.json"), "w") as fp:
            json.dump(records, fp, indent=4)

    for rank in range(args.num_gpus):
        payload_path = _worker_payload_path(args.output_dir, rank)
        if os.path.exists(payload_path):
            os.remove(payload_path)

    return results


if __name__ == "__main__":
    torch._dynamo.config.cache_size_limit = 128

    args = collect_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but is not available. Falling back to CPU.")
        args.device = "cpu"

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

    if args.device == "cuda" and args.num_gpus > 1:
        available_gpus = torch.cuda.device_count()
        if available_gpus < args.num_gpus:
            raise ValueError(f"Requested {args.num_gpus} GPUs, but only {available_gpus} are visible.")

        logger.info(f"Launching multi-GPU evaluation across {args.num_gpus} GPUs.")
        worker_args = vars(args).copy()
        worker_args.pop("logger", None)
        mp.spawn(_eval_worker, args=(args.num_gpus, worker_args), nprocs=args.num_gpus, join=True)
        results = _merge_worker_payloads(args)
        logger.info("\nEvaluation results:\n" + "\n".join("{} {:.3f}".format(k, v) for k, v in results.items()))
        logger.info("End of the evaluation")
        raise SystemExit(0)

    model_wrapped = get_model(args=args, device=args.device)

    # total_params = sum(p.numel() for p in model_wrapped.parameters())
    # args.logger.info(f"Total number of parameters: {total_params/1e6:.2f}M")

    if args.model_path != "original_pretrained":
        model_wrapped.load_from_pretrained(model_path=args.model_path, device=args.device)

    dataset_image_processor = getattr(
        model_wrapped, "image_processor_callable", getattr(model_wrapped, "image_processor", None)
    )
    dataset = get_dataset(args, dataset_image_processor)

    eval_engine = get_eval_engine(args=args, dataset=dataset)
    eval_engine.evaluate(args=args, model=model_wrapped)

    args.logger.info("End of the evaluation")
