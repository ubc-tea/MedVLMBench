# import csv
# import json
# import logging
# import os
# import re
# import string

# import numpy as np
# import pandas as pd
# import requests
# import torch
# from torch.utils.data import DataLoader

# from evaluation.metrics import MetricLogger


# class Benchmark:
#     def __init__(self, args, dataset, logger):
#         """Initialize the benchmark.

#         Args:
#             logger: A logger object.
#         """
#         # TODO: can replace logger with args.logger
#         self.task: str = "None"
#         self.prompt_template = "{}"
#         self.dataset = dataset
#         self.args = args
#         # self.modality = self.dataset.get_modality()

#         self.metric_logger = MetricLogger(delimiter=" ")
#         self.logger = logger

#     def evaluate(self, eval_args, model):
#         data_loader = DataLoader(self.dataset, batch_size=1)

#         with torch.inference_mode():
#             for batch in self.metric_logger.log_every(data_loader, eval_args.print_freq, header="Test:"):
#                 subject = [x[0] for x in batch]
#                 self.evaluate_subject(subject, model)

#         self.metric_logger.synchronize_between_processes()

#         return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}

#     def evaluate_subject(self, subject, model):
#         pass

#     def __len__(self):
#         """Get the length of the dataset."""
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         """Get an item from the dataset.

#         Args:
#             idx: The index of the item to get.

#         Returns:
#             The item from the dataset.
#         """
#         return {"idx": idx, "sample": self.dataset[idx]}

#     def save(self, path):
#         pass
