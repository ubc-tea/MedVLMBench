import nltk

nltk.download("wordnet")

from torchvision.transforms.functional import to_pil_image
from torchmetrics.functional.text import bleu_score, rouge_score
from nltk.translate.meteor_score import meteor_score

from eval.base import EvalEngine
from eval.metrics import calculate_exactmatch, calculate_bertscore
from eval.utils import normalize_word


class CaptionEvalEngine(EvalEngine):
    def __init__(self, args, dataset, logger):
        super().__init__(args, dataset, logger)

        self.task = "caption"

    def evaluate_subject(self, subject, model):
        image = subject["image"]
        caption = subject["label"]
        prompt_template = subject["prompt_template"]
        image_size = subject["image_size"]
        image_path = subject["image_path"]
        query = subject.get("query", "")

        caption_l = caption.lower()

        device = self.args.device
        image = image.to(device, non_blocking=True)

        prompt = prompt_template.format("").strip()
        output = model.infer_vision_language(image, prompt, image_size=image_size)
        output_l = output.lower()

        output_normed = normalize_word(output_l)
        caption_normed = normalize_word(caption_l)

        metrics = [
            "bleu1",
            "bleu2",
            "bleu3",
            "bleu4",
            "rouge1_fmeasure",
            "rouge1_precision",
            "rouge1_recall",
            "rouge2_fmeasure",
            "rouge2_precision",
            "rouge2_recall",
            "rougeL_fmeasure",
            "rougeL_precision",
            "rougeL_recall",
            "methor",
            "exact_match",
            "bertscore",
            # "bert_score_precision",
            # "bert_score_recall",
            # "bert_score_f1",
        ]

        exact_match = calculate_exactmatch(output_l, caption_l)

        bleu1 = bleu_score([output_normed], [[caption_normed]], n_gram=1).item()
        bleu2 = bleu_score([output_normed], [[caption_normed]], n_gram=2).item()
        bleu3 = bleu_score([output_normed], [[caption_normed]], n_gram=3).item()
        bleu4 = bleu_score([output_normed], [[caption_normed]], n_gram=4).item()
        rouge_scores = rouge_score(output_normed, caption_normed)
        methor = meteor_score([caption_normed.split()], output_normed.split())
        bertscore = calculate_bertscore(output_normed, caption_normed)
        # bert_scores = bert_score([output_normed], [caption_normed])

        for metric in metrics:
            if metric.startswith("rouge"):
                self.metric_logger.meters[metric].update(rouge_scores[metric].item(), n=1)
            # elif metric.startswith("bert_score"):
            #     self.metric_logger.meters[metric].update(bert_scores[metric.replace("bert_score_", "")].item(), n=1)
            else:
                self.metric_logger.meters[metric].update(eval(metric), n=1)

        if self.args.save_pred:
            self.records.append({"image_path": image_path, "caption": caption, "prediciton": output})
