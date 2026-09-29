def get_eval_engine(args, dataset):
    if args.task == "diagnosis":
        from eval.diagnosis import DiagnosisEvalEngine
        engine = DiagnosisEvalEngine
    elif args.task == "vqa":
        from eval.vqa import VQAEvalEngine
        engine = VQAEvalEngine
    elif args.task == "caption":
        from eval.caption import CaptionEvalEngine
        engine = CaptionEvalEngine
    else:
        raise ValueError(f"Unsupported evaluation task: {args.task}")
    return engine(args=args, dataset=dataset, logger=args.logger)
