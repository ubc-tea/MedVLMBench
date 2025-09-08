from typing import Any
from trl import SFTConfig, SFTTrainer
from torchvision.transforms.functional import to_pil_image


class MedGemmaTrainer:
    def __init__(self, args, model, dataset) -> None:
        self.args = args

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
                    model.processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False).strip()
                )

            # Tokenize the texts and process the images
            batch = model.processor(text=texts, images=images, return_tensors="pt", padding=True)

            # The labels are the input_ids, with the padding and image tokens masked in
            # the loss computation
            labels = batch["input_ids"].clone()

            # Mask image tokens
            image_token_id = [
                model.processor.tokenizer.convert_tokens_to_ids(
                    model.processor.tokenizer.special_tokens_map["boi_token"]
                )
            ]
            # Mask tokens that are not used in the loss computation
            labels[labels == model.processor.tokenizer.pad_token_id] = -100
            labels[labels == image_token_id] = -100
            labels[labels == 262144] = -100

            batch["labels"] = labels
            return batch

        self.sft_cfg = SFTConfig(
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

        self.trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            peft_config=model.peft_config,
            processing_class=model.processor,
            data_collator=collate_fn,
        )

    def train(self):
        self.trainer.train()


# def collate_fn()
