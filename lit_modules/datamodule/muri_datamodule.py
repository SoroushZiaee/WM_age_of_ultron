import os
from typing import List, Optional
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from argparse import Namespace
import torch

from datasets.Muri import MuriDataset


class MuriDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.data_dir = self.hparams.data_dir
        self.image_size = self.hparams.image_size
        self.batch_size = self.hparams.batch_size
        self.num_workers = self.hparams.num_workers
        self.change_labels = self.hparams.change_labels
        self.pin_memory_train, self.pin_memory_val, self.pin_memory_test = (
            self.hparams.pin_memories
        )
        self.return_paths = self.hparams.return_paths

        self.dims = (3, self.image_size, self.image_size)

        self.task_type = "regression"  # regression task

    def prepare_data(self):
        # Check if the data is already downloaded
        if not os.path.exists(os.path.join(self.data_dir, "images")):
            print("Muri data not found. Please download the dataset manually.")

    def setup(self, stage: Optional[str] = None):

        if stage == "fit" or stage is None:
            self.train_dataset = self.get_dataset(
                transforms=self.train_transform(self.image_size)
            )
            self.val_dataset = self.get_dataset(
                transforms=self.val_transform(self.image_size)
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.get_dataset(self.val_transform(self.image_size))

        if stage == "fit" or stage is None:
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            print(f"Test dataset size: {len(self.test_dataset)}")

    def get_dataset(self, transforms):
        return MuriDataset(
            root=self.data_dir, transforms=transforms, return_paths=self.return_paths
        )

    def train_transform(self, input_size: int = 256):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def val_transform(self, input_size: int = 256):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_val,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory_test,
        )

    def log_samples_to_tensorboard(self, logger):
        if self.task_type == "classification" or self.task_type == "combined":
            # Get a batch of data
            batch = next(iter(self.train_dataloader()))
            images, labels = batch
            if self.task_type == "combined":
                images, labels = images["classification"], labels["classification"]

            # Create a grid of images
            grid = torchvision.utils.make_grid(images)
            logger.experiment.add_image("sample_images", grid, 0)

            # Log labels
            if self.task_type == "classification":
                class_names = [f"Class_{i}" for i in range(self.num_classes)]
                # label_names = [class_names[label] for label in labels]
                logger.experiment.add_text("sample_labels", ", ".join(class_names), 0)
            elif self.task_type == "combined":
                logger.experiment.add_text(
                    "sample_classification_labels", str(labels.tolist()), 0
                )

        if self.task_type == "regression" or self.task_type == "combined":
            batch = next(iter(self.train_dataloader()))
            images, labels = batch
            if self.task_type == "combined":
                images, labels = images["regression"], labels["regression"]

            # Create a grid of images
            grid = torchvision.utils.make_grid(images)
            logger.experiment.add_image("sample_regression_images", grid, 0)

            # Log labels
            logger.experiment.add_text(
                "sample_regression_labels", str(labels.tolist()), 0
            )
