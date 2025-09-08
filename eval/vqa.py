from torchvision.transforms.functional import to_pil_image
from torchmetrics.functional.text import bleu_score, rouge_score

from eval.base import EvalEngine
from eval.metrics import (
    calculate_exactmatch,
    calculate_f1score,
    calculate_appearance_with_normalization,
    calculate_bertscore,
    calculate_meteor,
)
from eval.utils import normalize_word


def process_tokens(text):
    tokenized_text = set(text.split())
    tokenized_text.discard("")
    return tokenized_text


class VQAEvalEngine(EvalEngine):
    def __init__(self, args, dataset, logger):
        super().__init__(args, dataset, logger)

        self.task = "vqa"

    def evaluate_subject(self, subject, model):
        # evaluation batch size is 1
        image = subject["image"]
        qs = subject["query"]
        answer = subject["label"]
        is_open = subject["is_open"]
        prompt_template = subject["prompt_template"]
        image_size = subject["image_size"]
        image_path = subject["image_path"]

        qs_l, answer_l = qs.lower(), answer.lower()

        device = self.args.device
        image = image.to(device, non_blocking=True)

        prompt = prompt_template.format(qs)
        output = model.infer_vision_language(image, prompt, image_size=image_size)
        output_l = output.lower()

        output_normed = normalize_word(output_l)
        answer_normed = normalize_word(answer_l)

        f1_score, precision, recall = calculate_f1score(output_l, answer_l)
        exact_match = calculate_exactmatch(output_l, answer_l)

        if is_open:
            # evaluation of open questions
            open_metrics = [
                "bleu1",
                "bleu2",
                "bleu3",
                "bleu4",
                "rouge1",
                "rouge2",
                "rougeL",
                "exact_match",
                "recall",
                "precision",
                "f1_score",
                "accuracy",
                "bertscore",
                "meteor",
            ]
            bleu1 = bleu_score([output_normed], [[answer_normed]], n_gram=1).item()
            bleu2 = bleu_score([output_normed], [[answer_normed]], n_gram=2).item()
            bleu3 = bleu_score([output_normed], [[answer_normed]], n_gram=3).item()
            bleu4 = bleu_score([output_normed], [[answer_normed]], n_gram=4).item()
            rouge_scores = rouge_score(output_normed, answer_normed)
            rouge1, rouge2, rougeL = (
                rouge_scores["rouge1_fmeasure"].item(),
                rouge_scores["rouge2_fmeasure"].item(),
                rouge_scores["rougeL_fmeasure"].item(),
            )
            # accuracy = calculate_appearance_with_normalization(output_l, answer_l)
            accuracy = float(recall >= 0.75)
            bertscore = calculate_bertscore(output_normed, answer_normed)
            meteor = calculate_meteor(output_normed, answer_normed)

            for metric in open_metrics:
                self.metric_logger.meters[f"{metric}_open"].update(eval(metric), n=1)

            if self.args.gpt_eval:
                pass
        else:
            closed_metrics = [
                "exact_match",
                "recall",
                "precision",
                "f1_score",
                "accuracy",
            ]
            accuracy = 1 if answer_l in output_l else 0

            for metric in closed_metrics:
                self.metric_logger.meters[f"{metric}_closed"].update(eval(metric), n=1)

        self.metric_logger.meters["exact_match_overall"].update(exact_match, n=1)
        self.metric_logger.meters["recall_overall"].update(recall, n=1)
        self.metric_logger.meters["precision_overall"].update(precision, n=1)
        self.metric_logger.meters["f1_overall"].update(f1_score, n=1)

        if self.args.save_pred:
            self.records.append(
                {
                    "image_path": image_path,
                    "question_type": "open" if is_open else "closed",
                    "qs": qs,
                    "answer": answer,
                    "prediction": output,
                }
            )
