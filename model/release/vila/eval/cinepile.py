import argparse
import itertools
import json
import os
from collections import defaultdict

import torch
from datasets import load_dataset
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger

PROMPT_TEMPLATE = """You will be provided with subtitles from a specific scene of a movie and a few frames from that scene. After going through the movie scene and seeing the frames, please answer the question that follows. The question will have five possible answers labeled A, B, C, D, and E, please try to provide the most probable answer in your opinion. Your output should be just one of A,B,C,D,E and nothing else.

**Subtitles:** \n{subtitles}\n\nQuestion: {question}"""

CATEGORY_MAPPING = {
    "Character and\nRelationship Dynamics": "crd",
    "Narrative and\nPlot Analysis": "npa",
    "Setting and\nTechnical Analysis": "sta",
    "Temporal": "temp",
    "Theme Exploration": "th",
}

ANSWER_MAPPING = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    video_dir = args.video_dir = DATASETS["cinepile"]["video_dir"]

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # TODO(zhijianl): This will be removed in the future
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)

    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    # Load data and chunk it
    data = load_dataset("tomg-group-umd/cinepile")["test"]
    instances = data.select(range(dist.rank(), len(data), dist.size()))

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        video = llava.Video(os.path.join(args.video_dir, instance["yt_clip_link"].split("watch?v=")[-1] + ".mp4"))

        question, choices = instance["question"], instance["choices"]
        labels = [chr(ord("A") + i) for i in range(len(choices))]
        for label, option in zip(labels, choices):
            question += f"\n- {label}) {option}"
        prompt = PROMPT_TEMPLATE.format(subtitles=instance["subtitles"], question=question)

        response = model.generate_content([video, prompt], generation_config=generation_config)
        choice = parse_choice(response, labels)
        outputs.append(
            {
                "question": question,
                "choice": choice,
                "target": ANSWER_MAPPING[instance["answer_key_position"]],
                "category": CATEGORY_MAPPING[instance["question_category"]],
            }
        )

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Run evaluation
    counts = defaultdict(lambda: {"match": 0, "total": 0})
    for output in outputs:
        for category in ["overall", output["category"]]:
            counts[category]["match"] += output["choice"] == output["target"]
            counts[category]["total"] += 1

    metrics = {}
    for category in ["overall"] + list(CATEGORY_MAPPING.values()):
        metrics[category] = counts[category]["match"] / max(counts[category]["total"], 1)
    io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
