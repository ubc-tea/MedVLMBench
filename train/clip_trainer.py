import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Trainer, TrainerCallback
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Optional
from eval import get_eval_engine
from dataset import get_dataset
from dataset.utils import DiagnosisDataCollator
from torch.profiler import profile, record_function, ProfilerActivity


class CustomCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        This method is called at the end of each epoch.
        """
        if self.trainer.current_epoch % 10 == 0:
            print(f"Running evaluation at epoch {self.trainer.current_epoch}...")
            metrics = self.trainer.eval_engine.evaluate(args=self.trainer.args, model=self.trainer.model)
            print(f"Evaluation metrics at epoch {self.trainer.current_epoch}: {metrics}")
        self.trainer.current_epoch += 1
        if self.trainer.profile:
            total_epoch_forward = self.trainer.epoch_forward_flops
            total_epoch_backward = self.trainer.epoch_backward_flops
            total_epoch_flops = total_epoch_forward + total_epoch_backward

            avg_epoch_forward = total_epoch_forward / self.trainer.batch_count if self.trainer.batch_count > 0 else 0
            avg_epoch_backward = total_epoch_backward / self.trainer.batch_count if self.trainer.batch_count > 0 else 0

            self.trainer.total_train_forward_flops += total_epoch_forward
            self.trainer.total_train_backward_flops += total_epoch_backward
            total_train_flops = self.trainer.total_train_forward_flops + self.trainer.total_train_backward_flops

            self.trainer.args.logger.info(
                f"Epoch {self.trainer.current_epoch} FLOPs: "
                f"Total Forward: {total_epoch_forward/1e9:.2f} GFLOPS, "
                f"Total Backward: {total_epoch_backward/1e9:.2f} GFLOPS, "
                f"Combined: {total_epoch_flops/1e9:.2f} GFLOPS "
                f"(Avg per batch: Forward: {avg_epoch_forward/1e9:.2f} GFLOPS, "
                f"Backward: {avg_epoch_backward/1e9:.2f} GFLOPS)"
            )

            self.trainer.args.logger.info(
                f"Total training FLOPs up to epoch {self.trainer.current_epoch}: "
                f"Forward: {self.trainer.total_train_forward_flops/1e9:.2f} GFLOPS, "
                f"Backward: {self.trainer.total_train_backward_flops/1e9:.2f} GFLOPS, "
                f"Combined: {total_train_flops/1e9:.2f} GFLOPS"
            )

            self.trainer.epoch_forward_flops = 0
            self.trainer.epoch_backward_flops = 0
            self.trainer.batch_count = 0
    
    def on_train_end(self, args, state, control, **kwargs):
        """
        This method is called after the full training is complete.
        """
        print("Training complete. Running final evaluation...")
        metrics = self.trainer.eval_engine.evaluate(args=self.trainer.args, model=self.trainer.model)
        print(f"Final evaluation metrics: {metrics}")


class CLIPLPTrainer(Trainer):
    def __init__(self, model, args, image_processor, train_dataset, eval_dataset, **kwargs):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )

        dataset = get_dataset(args=args, image_processor_callable=getattr(model, "image_processor"), split="test") # safeguard that it is really testset here
        self.eval_engine = get_eval_engine(args=args, dataset=dataset)
        self.image_processor = image_processor
        self.current_epoch = 0

        # Accumulators for per-epoch FLOPs
        self.epoch_forward_flops = 0
        self.epoch_backward_flops = 0
        self.batch_count = 0
        
        # Global accumulators for the entire training process
        self.total_train_forward_flops = 0
        self.total_train_backward_flops = 0
        self.profile = False

        self.add_callback(CustomCallback(self))

    def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
        """
        Computes loss while tracking only forward FLOPS using torch.profiler.

        Args:
            model: The PyTorch model.
            inputs: Dictionary containing pixel_values and labels.
            num_items_in_batch: Number of items in batch (for normalization if needed).
            return_outputs: Whether to return logits with loss.

        Returns:
            Loss value (and logits if return_outputs=True).
        """
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]

        if self.image_processor is not None and pixel_values.dtype == torch.uint8:
            pixel_values = self.image_processor(pixel_values)

        def _forward_once():
            out = model(pixel_values)
            cls_name = out.__class__.__name__
            if isinstance(out, torch.Tensor):
                logits = out
                class_weights = self.train_dataset.class_weights.to(
                    logits.device, logits.dtype
                )
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                return loss, logits
            elif hasattr(out, "loss"):
                loss = out.loss
                if hasattr(out, "logits_per_image"):
                    logits = out.logits_per_image
                else:
                    raise ValueError(
                        "Model output does not contain logits_per_image."
                    )
                return loss, logits
            else:
                raise TypeError(
                    f"Model forward returned {type(out)} — expected Tensor or output with loss."
                )
        

        if self.profile:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=True,
            ) as prof:
                with record_function("forward_pass"):
                    loss, logits = _forward_once()
            self.epoch_forward_flops += sum(
                evt.flops for evt in prof.key_averages() if evt.flops is not None
            )
        else:
            loss, logits = _forward_once()

        return (loss, logits) if return_outputs else loss
    
    
    def training_step(self, model, inputs, *_args):
        """
        Runs a single training step, tracking FLOPS for both forward and backward passes.
        Hugging Face's Trainer calls this with (model, inputs, num_items_in_batch),
        so we use *_args to ignore additional parameters.
        """
        loss = self.compute_loss(model, inputs, num_items_in_batch=len(inputs))

        # Profile Backward Pass Separately
        if self.profile:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                profile_memory=True
            ) as prof:
                with record_function("backward_pass"):
                    loss.backward()  # Backward happens here, so we track FLOPS here

            # Extract Backward FLOPS
            flops_backward = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
            self.epoch_backward_flops += flops_backward

        else:
            loss.backward()
        self.batch_count += 1

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))
        # print(f"🔹 Backward FLOPS per batch: {flops_backward / 1e9:.2f} GFLOPS")

        return loss
    
    def get_labels(self, eval_preds):
        logits, labels = eval_preds
        return labels

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        
        model_to_save = self.model.model

        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module

        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))



def make_diagnosis_data_module(train_dataset, eval_dataset=None):

    data_collator = DiagnosisDataCollator()

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)



class CLIPTrainer(Trainer):
    def __init__(self, 
                 model, 
                 args, 
                 tokenizer, 
                 image_processor, 
                 train_dataset, 
                 eval_dataset, 
                 temperature=0.07, 
                 **kwargs):
        super().__init__(
            model=model,
            args=args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.image_processor = image_processor
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        image_embeds = outputs.get('image_embeds')
        text_embeds = outputs.get('text_embeds')

        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        return super().get_eval_dataloader(eval_dataset)