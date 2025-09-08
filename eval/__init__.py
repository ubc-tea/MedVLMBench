from eval.vqa import VQAEvalEngine
from eval.caption import CaptionEvalEngine
from eval.diagnosis import DiagnosisEvalEngine


task_engines = {"vqa": VQAEvalEngine, "caption": CaptionEvalEngine, "diagnosis": DiagnosisEvalEngine}


def get_eval_engine(args, dataset):
    engine = task_engines[args.task](args=args, dataset=dataset, logger=args.logger)
    return engine
