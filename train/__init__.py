from train.caption import CaptionTrainEngine
from train.vqa import VQATrainEngine
from train.lp import DiagnosisLPTrainEngine
from train.clip_trainer import CLIPLPTrainer, make_diagnosis_data_module
from train.clip_trainer import make_diagnosis_data_module

task_engines = {"vqa": VQATrainEngine, "diagnosis": DiagnosisLPTrainEngine, "caption": CaptionTrainEngine}


def get_trainer(args, model_wrapped, dataset):
    if args.model in ["LLaVA-1.5", "LLaVA-Med"]:
        from model.release.llava.train.llava_trainer import LLaVATrainer
        from train.llava_trainer import make_supervised_data_module

        data_module = make_supervised_data_module(
            args,
            dataset=dataset,
            tokenizer=model_wrapped.tokenizer,
            image_processor=model_wrapped.image_processor,
            model_constants=model_wrapped.constants,
        )
        trainer = LLaVATrainer(model=model_wrapped.model, args=args, tokenizer=model_wrapped.tokenizer, **data_module)

        return trainer
    elif args.model in ["MedGemma"]:
        from trl import SFTConfig, SFTTrainer
        from torchvision.transforms.functional import to_pil_image
        from typing import Any

        def collate_fn(subjects: list[dict[str, Any]]):
            texts = []
            images = []

            for subject in subjects:
                image = subject["image"]
                qs = subject["query"]
                answer = subject["label"]
                prompt_template = subject["prompt_template"]

                prompt = prompt_template.format(qs)

                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert on understanding medical images."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": answer,
                            },
                        ],
                    },
                ]

                images.append([to_pil_image(image).convert("RGB")])
                texts.append(
                    model_wrapped.processor.apply_chat_template(
                        messages, add_generation_prompt=False, tokenize=False
                    ).strip()
                )

            # Tokenize the texts and process the images
            batch = model_wrapped.processor(text=texts, images=images, return_tensors="pt", padding=True)

            # The labels are the input_ids, with the padding and image tokens masked in
            # the loss computation
            labels = batch["input_ids"].clone()

            # Mask image tokens
            image_token_id = [
                model_wrapped.processor.tokenizer.convert_tokens_to_ids(
                    model_wrapped.processor.tokenizer.special_tokens_map["boi_token"]
                )
            ]
            # Mask tokens that are not used in the loss computation
            labels[labels == model_wrapped.processor.tokenizer.pad_token_id] = -100
            labels[labels == image_token_id] = -100
            labels[labels == 262144] = -100

            batch["labels"] = labels
            return batch

        sft_cfg = SFTConfig(
            output_dir=args.output_dir,  # Directory and Hub repository id to save the model to
            num_train_epochs=args.num_train_epochs,  # Number of training epochs
            per_device_train_batch_size=args.per_device_train_batch_size,  # Batch size per device during training
            per_device_eval_batch_size=args.per_device_eval_batch_size,  # Batch size per device during evaluation
            gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of steps before performing a backward/update pass
            gradient_checkpointing=args.gradient_checkpointing,  # Enable gradient checkpointing to reduce memory usage
            optim=args.optim,  # Use fused AdamW optimizer for better performance
            logging_steps=args.logging_steps,  # Number of steps between logs
            save_strategy=args.save_strategy,  # Save checkpoint every epoch
            eval_strategy="no",  # Evaluate every `eval_steps`
            # eval_steps=50,  # Number of steps between evaluations
            learning_rate=args.learning_rate,  # Learning rate based on QLoRA paper
            bf16=args.bf16,  # Use bfloat16 precision
            max_grad_norm=0.3,  # Max gradient norm based on QLoRA paper
            warmup_ratio=args.warmup_ratio,  # Warmup ratio based on QLoRA paper
            lr_scheduler_type=args.lr_scheduler_type,  # Use linear learning rate scheduler
            push_to_hub=False,  # Push model to Hub
            report_to="none",  # Report metrics to tensorboard
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },  # Set gradient checkpointing to non-reentrant to avoid issues
            dataset_kwargs={"skip_prepare_dataset": True},  # Skip default dataset preparation to preprocess manually
            remove_unused_columns=False,  # Columns are unused for training but needed for data collator
            label_names=["labels"],  # Input keys that correspond to the labels
        )

        trainer = SFTTrainer(
            model=model_wrapped.model,
            args=sft_cfg,
            train_dataset=dataset,
            peft_config=model_wrapped.peft_config,
            processing_class=model_wrapped.processor,
            data_collator=collate_fn,
        )

        return trainer
    elif args.model in ["NVILA", "VILA1.5", "VILA-M3"]:
        from model.release.vila.train.llava_trainer import LLaVATrainer
        from model.release.vila.data import make_supervised_data_module

        data_module = make_supervised_data_module(
            args,
            dataset=dataset,
            tokenizer=model_wrapped.tokenizer,
            image_processor=model_wrapped.image_processor,
            model_constants=model_wrapped.constants,
        )
        trainer = LLaVATrainer(model=model_wrapped.model, args=args, tokenizer=model_wrapped.tokenizer, **data_module)

        return trainer
    elif args.model in ["Lingshu"]:
        from transformers import Trainer
        from model.lingshu import LingshuDataset, LingshuCollator

        ds_train = LingshuDataset(args, dataset, model_wrapped.processor)

        collator = LingshuCollator(
            pad_token_id=model_wrapped.processor.tokenizer.pad_token_id,
            ignore_index=model_wrapped.tokenizer.pad_token_id,
        )
        trainer = Trainer(
            model=model_wrapped.model,
            args=args,
            tokenizer=model_wrapped.tokenizer,
            train_dataset=ds_train,
            data_collator=collator,
        )

        return trainer
    
    elif args.model in ["CLIP", "MedCLIP", "PMCCLIP", "PLIP", "MedSigLIP", "XrayGPT", "BioMedCLIP", "BLIP", "BLIP2-2.7b", "PubMedCLIP", "SigLIP"]:        
        if args.usage in ["lp", "img-lora-lp", "clip-img-lora"]:
            data_module = make_diagnosis_data_module(
                train_dataset=dataset,
            )

            trainer = CLIPLPTrainer(
                model=model_wrapped, args=args, image_processor=model_wrapped.image_processor, **data_module
            )

            return trainer
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError("Trainer not supported for {}".format(args.model))


def get_train_engine(args, model_wrapped, dataset):
    engine = task_engines[args.task](
        args=args,
        dataset=dataset,
        model_wrapped=model_wrapped,
        logger=args.logger,
        hf_trainer=get_trainer(args, model_wrapped=model_wrapped, dataset=dataset),
    )

    return engine
