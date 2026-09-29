from train.base import TrainEngine


class DiagnosisLPTrainEngine(TrainEngine):
    def __init__(self, args, dataset, model_wrapped, logger, hf_trainer=None):
        super().__init__(args, dataset, model_wrapped, logger, hf_trainer)

        self.task = "lp"

    def save(self):
        self.hf_trainer.save_state()
        self.hf_trainer.save_model(self.args.output_dir)
