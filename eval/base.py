import csv
import json
import logging
import os
import string
import json

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, Subset

from eval.metrics import MetricLogger


class EvalEngine:
    def __init__(self, args, dataset, logger):
        """Initialize the benchmark.

        Args:
            logger: A logger object.
        """
        self.args = args
        self.task: str = "None"
        self.prompt_template = "{}"
        self.dataset = dataset

        self.metric_logger = MetricLogger(logger=logger, delimiter=" ")
        self.records = []
        self.logger = logger

    def evaluate(self, args, model, indices=None, save_outputs=True):
        # Keep samples intact so metadata like image_paths (possibly multi-image) is preserved.
        dataset = self.dataset if indices is None else Subset(self.dataset, indices)
        max_samples = getattr(args, "max_samples", None)
        if max_samples is not None:
            dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
        data_loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])
        header = getattr(args, "eval_header", "Test:")

        self.init_metric_logger()

        with torch.inference_mode():
            for subject in self.metric_logger.log_every(data_loader, args.eval_print_freq, header=header):
                try:
                    self.evaluate_subject(subject, model)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    self.logger.exception("Skipping sample due to evaluation error: %s", exc)
                    self.record_failure(subject, exc)

        self.metric_logger.synchronize_between_processes()

        if save_outputs:
            self.save(self.args.output_dir, model)

        results = {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}

        self.logger.info("\nEvaluation results:\n" + "\n".join("{} {:.3f}".format(k, v) for k, v in results.items()))

        return results

    def export_state(self, model):
        return {
            "model_name": model.name,
            "model_type": model.model_type,
            "dataset_name": self.dataset.name,
            "dataset_modality": self.dataset.modality,
            "records": self.records,
            "meters": {
                name: {
                    "total": meter.total,
                    "count": meter.count,
                }
                for name, meter in self.metric_logger.meters.items()
            },
        }

    def evaluate_subject(self, subject, model):
        pass

    def init_metric_logger(self):
        self.metric_logger = MetricLogger(logger=self.logger, delimiter=" ")

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):  # TODO: Check why this method is implemented here?
        """Get an item from the dataset.

        Args:
            idx: The index of the item to get.

        Returns:
            The item from the dataset.
        """
        return {"idx": idx, "sample": self.dataset[idx]}

    def save(self, path, model):
        info = {
            "model": [model.name],
            "model_name": [os.path.basename(self.args.model_path)],
            "task": [self.task],
            "dataset": [self.dataset.name],
            "model_type": [model.model_type],
            "modality": [self.dataset.modality],
            "size": [len(self.dataset)],
        }
        info = {**info, **{k: [meter.global_avg] for k, meter in self.metric_logger.meters.items()}}

        df = pd.DataFrame(info)
        df.to_csv(os.path.join(path, "results.csv"), index=False)

        if self.args.save_pred:
            with open(os.path.join(path, "predictions.json"), "w") as fp:
                json.dump(self.records, fp, indent=4)

    def record_failure(self, subject, exc):
        if not self.args.save_pred:
            return

        def _safe_value(value):
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                return [_safe_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _safe_value(v) for k, v in value.items()}
            return str(value)

        record = {
            "image_path": _safe_value(subject.get("image_path")),
            "prediction": "",
            "error": str(exc),
            "status": "failed",
        }

        for key in ["question_type", "query", "qs", "label", "answer", "caption", "prompt_template", "image_paths"]:
            if key in subject:
                record[key] = _safe_value(subject.get(key))

        self.records.append(record)
