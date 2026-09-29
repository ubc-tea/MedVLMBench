import json
import tempfile
import unittest
from argparse import Namespace

try:
    import torch
    from utils.experiment_tracking import ExperimentTracker
except ModuleNotFoundError:
    torch = None

try:
    from dataset import FractionalDataset, _apply_train_fraction
except ModuleNotFoundError:
    FractionalDataset = None
    _apply_train_fraction = None


@unittest.skipIf(torch is None, "project PyTorch environment is not available")
class RevisionInfrastructureTests(unittest.TestCase):
    @unittest.skipIf(FractionalDataset is None, "project dataset dependencies are not available")
    def test_fraction_is_deterministic_and_manifested(self):
        class DummyDataset(torch.utils.data.Dataset):
            name = "dummy"
            modality = "test"
            split = "train"

            def __len__(self):
                return 10

            def __getitem__(self, index):
                return {"index": index}

        with tempfile.TemporaryDirectory() as output_dir:
            args = Namespace(
                train_fraction=0.3,
                fraction_seed=7,
                output_dir=output_dir,
                dataset="dummy",
            )
            first = _apply_train_fraction(DummyDataset(), args, "train")
            second = _apply_train_fraction(DummyDataset(), args, "train")
            self.assertIsInstance(first, FractionalDataset)
            self.assertEqual(len(first), 3)
            self.assertEqual(first.indices, second.indices)
            with open(f"{output_dir}/train_subset_manifest.json") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["selected_size"], 3)
            self.assertEqual(manifest["indices"], first.indices)
            self.assertEqual(len(_apply_train_fraction(DummyDataset(), args, "test")), 10)

    def test_tracker_records_parameters_and_resources(self):
        class DummyDataset(torch.utils.data.Dataset):
            name = "dummy"
            modality = "test"
            split = "train"

            def __len__(self):
                return 10

            def __getitem__(self, index):
                return {"index": index}

        class DummyModel(torch.nn.Module):
            name = "dummy-model"
            model_type = "general"

            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1)

        with tempfile.TemporaryDirectory() as output_dir:
            args = Namespace(model="dummy-model", model_path="checkpoint", dataset="dummy", split="test")
            tracker = ExperimentTracker(output_dir, "evaluation", args, DummyModel(), DummyDataset())
            tracker.start()
            tracker.finish("completed")
            with open(f"{output_dir}/experiment_manifest.json") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["model"]["parameters"]["total"], 3)
            self.assertEqual(manifest["dataset"]["n_samples"], 10)
            self.assertIn("wall_time_seconds", manifest["resources"])

    def test_training_manifest_counts_actual_microbatches_and_final_parameters(self):
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 3

            def __getitem__(self, index):
                return index

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Linear(2, 2, bias=False)
                self.adapter = torch.nn.Linear(2, 1, bias=False)
                self.frozen.requires_grad_(False)

        class DummyTrainer:
            def __init__(self):
                self.model = DummyModel()
                self.train_dataset = DummyDataset()
                self.state = Namespace(global_step=1, epoch=1.0, total_flos=123.0)

            def training_step(self, model, inputs):
                return inputs["input_ids"].sum()

        with tempfile.TemporaryDirectory() as output_dir:
            args = Namespace(
                model="dummy", model_path="checkpoint", dataset="dummy",
                split="train", world_size=1, num_train_epochs=1,
                per_device_train_batch_size=2, gradient_accumulation_steps=1,
            )
            trainer = DummyTrainer()
            tracker = ExperimentTracker(output_dir, "train", args, trainer.model, DummyDataset())
            tracker.start()
            tracker.attach_trainer(trainer)
            trainer.training_step(trainer.model, {
                "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
                "labels": torch.tensor([[-100, 2, -100], [-100, -100, -100]]),
                "images": torch.zeros(2, 3, 4, 4),
            })
            tracker.finish("completed")
            with open(f"{output_dir}/experiment_manifest.json") as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["dataset"]["trainer_n_samples"], 3)
            self.assertEqual(manifest["model"]["parameters"]["total"], 6)
            self.assertEqual(manifest["model"]["parameters"]["trainable"], 2)
            self.assertEqual(manifest["training"]["observed"]["input_tokens"], 3)
            self.assertEqual(manifest["training"]["observed"]["padded_input_tokens"], 6)
            self.assertEqual(manifest["training"]["observed"]["supervised_tokens"], 1)
            self.assertEqual(manifest["training"]["observed"]["images_seen"], 2)
            self.assertEqual(manifest["training"]["token_count_scope"], "full_run")
            self.assertEqual(manifest["dataset"]["fraction_realized"], 1.0)

    def test_tracker_records_training_and_inference_flops_per_sample(self):
        class DummyDataset(torch.utils.data.Dataset):
            name = "dummy"
            modality = "test"
            split = "train"

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return index

        class DummyTrainer:
            def __init__(self):
                self.model = torch.nn.Linear(4, 2)
                self.train_dataset = DummyDataset()
                self.state = Namespace(global_step=1, epoch=1.0, total_flos=0.0)

            def training_step(self, model, inputs):
                output = model(inputs["pixel_values"])
                loss = output.square().mean()
                loss.backward()
                return loss.detach()

        with tempfile.TemporaryDirectory() as output_dir:
            args = Namespace(
                model="dummy", model_path="checkpoint", dataset="dummy", split="train",
                world_size=1, num_train_epochs=1, per_device_train_batch_size=2,
                gradient_accumulation_steps=1, flops_profile_batches=1,
            )
            trainer = DummyTrainer()
            tracker = ExperimentTracker(output_dir, "train", args, trainer.model, DummyDataset())
            tracker.start()
            tracker.attach_trainer(trainer)
            trainer.training_step(trainer.model, {
                "pixel_values": torch.ones(2, 4),
                "labels": torch.zeros(2, dtype=torch.long),
            })
            with tracker.profile_inference(sample_count=2):
                trainer.model(torch.ones(2, 4))
            tracker.finish("completed")
            with open(f"{output_dir}/experiment_manifest.json") as stream:
                manifest = json.load(stream)
            flops = manifest["training"]["flops"]
            self.assertEqual(flops["status"], "measured")
            self.assertEqual(flops["profiled_samples"], 2)
            self.assertGreater(flops["flops_per_sample"], 0)
            self.assertEqual(
                manifest["training"]["estimated_profiled_operator_flops_full_run"],
                flops["flops_per_sample"] * manifest["training"]["observed"]["examples_seen"],
            )
            self.assertGreater(manifest["inference"]["flops"]["flops_per_sample"], 0)

        with tempfile.TemporaryDirectory() as output_dir:
            args = Namespace(
                model="dummy", model_path="checkpoint", dataset="dummy", split="test",
                flops_profile_batches=1,
            )
            model = torch.nn.Linear(4, 2)
            tracker = ExperimentTracker(output_dir, "evaluation", args, model, DummyDataset())
            tracker.start()
            with tracker.profile_inference(sample_count=2):
                model(torch.ones(2, 4))
            tracker.finish("completed")
            with open(f"{output_dir}/experiment_manifest.json") as stream:
                manifest = json.load(stream)
            flops = manifest["inference"]["flops"]
            self.assertEqual(flops["status"], "measured")
            self.assertEqual(flops["profiled_samples"], 2)
            self.assertGreater(flops["flops_per_sample"], 0)

    def test_diagnosis_checkpoint_contains_linear_probe_head(self):
        from train.clip_trainer import CLIPLPTrainer

        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 2)

        with tempfile.TemporaryDirectory() as output_dir:
            trainer = Namespace(model=Probe(), args=Namespace(output_dir=output_dir))
            CLIPLPTrainer.save_model(trainer)
            state = torch.load(f"{output_dir}/pytorch_model.bin", weights_only=True)
            self.assertIn("head.weight", state)
            self.assertIn("model.weight", state)

            from utils.experiment_tracking import _parameter_counts
            counts = _parameter_counts(trainer.model)
            self.assertEqual(counts["total"], 12)
            self.assertEqual(counts["trainable"], 12)

if __name__ == "__main__":
    unittest.main()
