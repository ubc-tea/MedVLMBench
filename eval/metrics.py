import os
import math
import time
from collections import defaultdict, deque
import datetime
import numpy as np

import torch
import torch.distributed as dist
from torch import inf
from transformers import AutoTokenizer, AutoConfig
from bert_score import score as bertscore
from nltk.translate.meteor_score import meteor_score, single_meteor_score

from eval.utils import normalize_word, split_sentence, brevity_penalty, modified_precision


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger(object):
    def __init__(self, logger, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.logger = logger
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable))
        )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def bleu(candidate, references, n, weights):

    pn = []
    bp = brevity_penalty(candidate, references)
    for i in range(n):
        pn.append(modified_precision(candidate, references, i + 1))
    if len(weights) > len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(weights[i])
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is bigger than n)"
    elif len(weights) < len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(0)
        for i in range(len(weights)):
            tmp_weights[i] = weights[i]
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return str(bleu_result) + " (warning: the length of weights is smaller than n)"
    else:
        bleu_result = calculate_bleu(weights, pn, n, bp)
        return str(bleu_result)


# BLEU
def calculate_bleu(weights, pn, n, bp):
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result


# Exact match
def calculate_exactmatch(candidate, reference):

    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0:
        return 0  # "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


# Exact match with normalization


def similarity_candidate_prediction(candidate_answer, prediction):

    candidate_answer = split_sentence(candidate_answer, 1)

    count = 0
    total = 0
    for word in prediction:
        if word in candidate_answer:
            count += 1

    total = len(candidate_answer)

    if total == 0:
        return 0.0  # "0 (warning: length of candidate's words is 0)"
    else:
        return count / total


def argmax(lst):
    return lst.index(max(lst))


def calculate_appearance_with_normalization(prediction, reference, candidate_set):

    prediction = normalize_word(prediction)
    reference = normalize_word(reference)
    prediction_words = split_sentence(prediction, 1)
    reference_words = split_sentence(reference, 1)

    candidate_set = candidate_set["0"]

    similarity_list = []
    candidate_answer_normalized_list = []
    for candidate_answer in candidate_set:

        if isinstance(candidate_answer, int):
            candidate_answer = str(candidate_answer)

        candidate_answer = normalize_word(candidate_answer)
        candidate_answer_normalized_list.append(candidate_answer)
        similarity_list.append(similarity_candidate_prediction(candidate_answer, prediction_words))

    final_prediction = candidate_answer_normalized_list[argmax(similarity_list)]

    # import pdb; pdb.set_trace()

    if final_prediction == reference:
        return 1.0  #
    else:
        return 0.0


# F1
def calculate_f1score(candidate, reference):

    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return 0, 0, 0  # "0 (warning: length of candidate's words is 0)"
    elif len(reference_words) == 0:
        return 0, 0, 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0, 0, 0
        else:
            return 2 * precision * recall / (precision + recall), precision, recall



_TOKENIZER_CACHE = {}

#: Any model_max_length or max_position_embeddings above this
#: is treated as “infinite / unknown”.
_SENTINEL = 10_000


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    """Load (or retrieve from cache) the tokenizer for `model_name`."""
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = tok
    return tok


def _model_window(model_name: str, tok: AutoTokenizer):
    """
    Return the usable context-window size for this model, or None if unlimited.

    Priority:
    1. tokenizer.model_max_length, if it looks reasonable (≤ _SENTINEL)
    2. model config's max_position_embeddings, if it looks reasonable
    3. None  → treat as “no limit known”
    """
    t_lim = getattr(tok, "model_max_length", None)
    if t_lim and t_lim < _SENTINEL:
        return int(t_lim)

    try:
        cfg = AutoConfig.from_pretrained(model_name)
        c_lim = getattr(cfg, "max_position_embeddings", None)
        if c_lim and c_lim < _SENTINEL:
            return int(c_lim)
    except Exception:
        # happens for trust_remote_code models, local dirs without config, etc.
        pass

    return None


def _trim(text: str, limit, tok: AutoTokenizer) -> str:
    """
    If `limit` is an int, truncate `text` so that
    (CLS) + tokens + (SEP) ≤ limit. Otherwise return `text` unchanged.
    """
    if limit is None:
        return text

    allowed = max(limit - 2, 1)
    ids = tok.encode(text, add_special_tokens=False)[:allowed]
    return tok.decode(ids, skip_special_tokens=True)


def calculate_bertscore(
    candidate: str,
    references: str,
    model_type: str = "emilyalsentzer/Bio_ClinicalBERT",
    device: str | None = None,
    reduction: str = "max",
):
    
    if bertscore is None:
        raise ImportError(
            "bert_score library not found. Install it with:  pip install bert-score"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tok = _get_tokenizer(model_type)
    limit = _model_window(model_type, tok)
 
    cand_proc = _trim(candidate, limit, tok)
    ref_proc  = _trim(references, limit, tok)

    P, R, F1 = bertscore(
        [cand_proc], [ref_proc],
        model_type=model_type,
        num_layers=12,
        device=device,
    )

    if reduction == "mean":
        return F1.mean().item()
    elif reduction == "max":
        return F1.max().item()
    else:
        raise ValueError("reduction must be either 'max' or 'mean'")


def calculate_meteor(
    candidate: str,
    references: str,
    alpha: float = 0.9,            # precision/recall balance; 0.9 is the default
    beta:  float = 3.0,            # fragmentation penalty weight
    gamma: float = 0.5,            # synonym/Stem penalty
) -> float:

    return single_meteor_score(
        [references], [candidate],
        alpha=alpha, beta=beta, gamma=gamma
    )