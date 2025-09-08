from torchmetrics.text import BLEUScore
from dataset.benchmark import clean_str, BatcherInput


class QA:
    """A benchmark for Question Answering tasks."""

    def __init__(self, **kwargs):
        """Initialize the QA task."""
        super().__init__(**kwargs)
        self.task = "QA"

    @abstractmethod
    def get_correct_answer(self, sample, full_text=False):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
            fullText: Returns the raw answer. Defaults to False.
        """

    @abstractmethod
    def get_predicted_answer(self, pred: str, sample):
        """Converts the free form text output to the answer index.

        Args:
            pred: The free form text output of the model.
            sample: The sample used to generate the answer.
        """

    def evaluate(self, predictions):
        """Evaluate the predictions against the ground truth.

        Args:
            predictions: The predictions made by the model.

        Returns:
            The evaluation output.
        """
        correct_answers = 0
        total_answers = 0

        answers_log = []

        for prediction in predictions:
            answer = prediction["answer"].text
            idx = prediction["idx"]
            sample = self[idx]["sample"]

            gold = self.get_correct_answer(sample)
            pred = self.get_predicted_answer(answer, sample)
            if pred == gold:
                correct_answers += 1
            total_answers += 1

            answers_log.append(
                (
                    self.get_correct_answer(sample, full_text=True),
                    answer,
                    pred,
                    gold,
                    pred == gold,
                )
            )

        metrics = {"accuracy": correct_answers / total_answers}

        return EvaluationOutput(answer_log=answers_log, metrics=metrics)


class MedQA(QA):
    """The MedQA task."""

    def __init__(self, **kwargs):
        """Initialize the MedQA task."""
        super().__init__(**kwargs)
        self.task_name = "MedQA"
        self.modality = "General medicine"
        self.bleu_scorer = BLEUScore(n_gram=1)

    def setup(self):
        """Setup the MedQA task family."""
        cache_dir = self.engine.get_config()["medqa_dir"]

        if cache_dir is None:
            raise ValueError("No path for MedQA dataset provided in the config file. Skipping the task.")

        self.dataset = load_dataset(
            "bigbio/med_qa",
            name="med_qa_en_source",
            split="test",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.train_dataset = load_dataset(
            "bigbio/med_qa",
            name="med_qa_en_source",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def format_question(self, sample, prompt=False):
        """Format the question for the MedQA task.

        Args:
            sample: The sample to format.
            prompt: Whether or not to add the answer in the prompt. Defaults to False.

        Returns:
            An instance of BatcherInput with The formatted question.
        """
        question = sample["question"]
        options = sample["options"]

        formatted_question = f"{question}\n"
        formatted_question += (
            "Options:\n" + "\n".join([f'{option["key"]}: {option["value"]}.' for option in options]) + "\n"
        )
        formatted_question += "What is the correct answer?"
        batcher_input = BatcherInput()

        batcher_input._add_text_prompt("user", formatted_question)

        # question = [{"role": "user", "content": formatted_question}]
        if prompt:
            formatted_answer = "The answer is " + sample["answer_idx"] + "."
            # question.append({"role": "assistant", "content": formatted_answer})
            batcher_input._add_text_prompt("assistant", formatted_answer)
        return batcher_input

    def get_correct_answer(self, sample, full_text=False):
        """Get the correct answer for the sample.

        Args:
            sample: The sample to get the correct answer for.
            fullText: Whether or not to return the raw text answer. Defaults to False.

        Returns:
            The correct answer.
        """
        if full_text:
            return f"{sample['answer_idx']}: {sample['answer'].lower().strip()}"

        return sample["answer_idx"].lower().strip()

    def get_predicted_answer(self, pred: str, sample):
        """Get the answer predicted by the model.

        Args:
            pred: The generated answer.
            sample: The sample used to generate the answer.

        Returns:
            The predicted answer.
        """
        pred = clean_str(pred)
        if len(pred) == 0:
            return "Invalid answer"

        options = [clean_str(f'{option["key"]} {option["value"]}') for option in sample["options"]]
        # Compute the BLEU score for each option
        scores = [self.bleu_scorer([pred], [[option]]) for option in options]

        if max(scores) == 0:
            return "Invalid answer"

        pred = sample["options"][scores.index(max(scores))]["key"].lower()

        return pred
