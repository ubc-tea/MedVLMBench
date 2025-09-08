from train.base import TrainEngine


class VQATrainEngine(TrainEngine):
    def __init__(self, args, dataset, model_wrapped, logger, hf_trainer=None):
        super().__init__(args, dataset, model_wrapped, logger, hf_trainer)

        self.task = "vqa"
