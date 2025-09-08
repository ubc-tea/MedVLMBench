import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from eval.base import EvalEngine
from dataset.utils import DiagnosisDataCollator

Metrics = namedtuple("Metrics", ["AUC", "ACC"])


class DiagnosisEvalEngine(EvalEngine):
    # TODO: support for multi-class
    def __init__(self, args, dataset, logger, task="binary-class", device="cuda"):
        super().__init__(args, dataset, logger)
        
        self.dataset = dataset
        self.logger = logger
        self.task = task.lower()
        if self.task not in {"multi-class", "binary-class"}:
            raise ValueError(f"Unsupported task: {task}")
        
        self.device = device 
        self.records = {
            "true_label": [],
            "pred_label": [],
            "pred_score": [],
        }

    def evaluate(self, args, model):
        """Run evaluation on the classification dataset."""
        args.logger.info("Length of the evaluation dataset: {}".format(len(self.dataset)))
        data_loader = DataLoader(self.dataset, batch_size=64, collate_fn=DiagnosisDataCollator(), shuffle=False)
        self.num_classes = model.num_classes
        self.init_metric_logger()
        self.records = {
            "true_label": [],
            "pred_label": [],
            "pred_score": [],
        }
        all_true = []
        all_out = []

        with torch.no_grad():
            for batch in self.metric_logger.log_every(data_loader, self.args.eval_print_freq, header="Test:"):
                t_np, o_np, acc = self._evaluate_batch(batch, model)
                bsz = len(t_np)
                all_true.append(t_np)
                all_out.append(o_np)
                self.metric_logger.meters["accuracy"].update(acc, n=bsz)

        true_np = np.concatenate(all_true, axis=0)
        out_np = np.concatenate(all_out, axis=0)
        
        auc_val = self._compute_auc(true_np, out_np)
        self.metric_logger.meters["auc"].update(auc_val, n=len(true_np))

        self.metric_logger.synchronize_between_processes()
        results = {k: m.global_avg for k, m in self.metric_logger.meters.items()}
        self.logger.info("\nEvaluation results:\n" + "\n".join(f"{k}: {v:.6f}" for k, v in results.items()))
        self.save(self.args.output_dir, model)

        return results

    def _evaluate_batch(self, batch, model):
        """Evaluate a single batch (image and label)."""
        image = batch["pixel_values"].to(self.device, non_blocking=True)
        true = batch["labels"].to(self.device)
        # image = model.image_processor_evaluation(image)

        model.to(self.device)

        out = model(image)
        
        # Convert logits → probabilities
        if out.size(-1) == 1:
            out = torch.sigmoid(out)
        else:
            out = torch.softmax(out, dim=-1)

        pred = out.argmax(dim=-1) if out.size(-1) > 1 else (out > 0.5).int()
        acc = self._batch_accuracy(true, pred)

        true_np = true.cpu().numpy()
        out_np = out.cpu().numpy()

        # For binary task ensure 2-column shape for AUC
        if self.task == "binary-class" and out_np.ndim == 1:
            out_np = np.vstack([1 - out_np, out_np]).T

        if self.args.save_pred:
            self.records["true_label"].extend(true_np.tolist())
            self.records["pred_label"].extend(pred.cpu().numpy().tolist())
            self.records["pred_score"].extend(out_np[:, 1].tolist())

        return true_np, out_np, acc

    def _batch_accuracy(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        """Compute accuracy for a single batch."""
        if true.ndim > 1:  # one-hot → class index
            true = true.argmax(dim=-1)
        return (true == pred).float().mean().item()

    def _compute_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Wrapper around *roc_auc_score* for binary & multi-class with better messages."""
        if self.task == "binary-class":
            if y_true.ndim > 1 and y_true.shape[1] == 2: # y_true.shape = (920,)
                y_true = y_true.argmax(axis=1)
            if y_score.ndim == 2 and y_score.shape[1] == 2:
                y_score = y_score[:, 1]
            return roc_auc_score(y_true, y_score)

    
    def save(self, path: str, model) -> None:
        """Save a one-row CSV with global averages and, optionally, raw predictions."""
        os.makedirs(path, exist_ok=True)
        info = {
            "task": [self.task],
            "dataset": [getattr(self.dataset, "name", "dataset")],
            "model_type": [getattr(model, "model_type", "")],
            "modality": [getattr(self.dataset, "modality", "image")],
            "size": [len(self.dataset)],
        }
        info |= {k: [m.global_avg] for k, m in self.metric_logger.meters.items()}
        pd.DataFrame(info).to_csv(os.path.join(path, "results.csv"), index=False)

        if self.args.save_pred:
            with open(os.path.join(path, "predictions.json"), "w", encoding="utf-8") as fp:
                json.dump(self.records, fp, indent=2)
