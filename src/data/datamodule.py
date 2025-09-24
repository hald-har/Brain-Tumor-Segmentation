"""
Data module for PyTorch image segmentation tasks.

Defines SegmentationDataModule to handle train/val/test/sample dataloaders
with configurable transforms and dataset paths.
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

from src.data.dataset import BaseDataset


class SegmentationDataModule:
    """
    PyTorch-style data module for image segmentation.

    Handles full dataset, train/val split, test dataset, and sample dataset.
    Allows custom image and mask transforms.

    Attributes:
        config (dict): Configuration loaded from YAML.
        dataset (BaseDataset): Full dataset.
        train_dataset (Subset): Training split.
        val_dataset (Subset): Validation split.
    """

    def __init__(self, config_path="configs/config.yaml", transforms=None, target_transforms=None):
        """Initialize the data module and datasets."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        # Dataset paths and training settings
        self.images_folder = self.config["dataset"]["images_folder"]
        self.masks_folder = self.config["dataset"]["masks_folder"]
        self.batch_size = self.config["training"]["batch_size"]
        self.num_workers = self.config["training"]["num_workers"]
        self.val_split = self.config["split"]["val_split"]
        self.shuffle = self.config["split"]["shuffle"]
        self.random_seed = self.config["split"]["random_seed"]
        self.image_size = self.config["dataset"]["image_size"]

        # Sample dataset settings
        self.sample_images_folder = self.config["sample_dataset"]["images_folder"]
        self.sample_masks_folder = self.config["sample_dataset"]["masks_folder"]
        self.sample_batch_size = self.config["sample_dataset"]["batch_size"]
        self.sample_num_workers = self.config["sample_dataset"]["num_workers"]

        # Default transforms
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
            ])
        else:
            self.transforms = transforms

        if target_transforms is None:
            self.target_transforms = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
            ])
        else:
            self.target_transforms = target_transforms

        # Full dataset and train/val split
        self.dataset = BaseDataset(
            self.images_folder, self.masks_folder, self.transforms, self.target_transforms
        )
        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def train_dataloader(self):
        """Returns DataLoader for training dataset."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Returns DataLoader for validation dataset."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Returns DataLoader for test dataset."""
        test_images_folder = self.config["dataset"]["test_images_folder"]
        test_masks_folder = self.config["dataset"]["test_masks_folder"]

        test_dataset = BaseDataset(
            test_images_folder,
            test_masks_folder,
            self.transforms,
            self.target_transforms,
        )

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def sample_train_dataloader(self):
        """Returns DataLoader for sample dataset used for quick testing."""
        sample_dataset = BaseDataset(
            self.sample_images_folder,
            self.sample_masks_folder,
            self.transforms,
            self.target_transforms,
        )

        val_size = int(len(sample_dataset) * self.val_split)
        train_size = len(sample_dataset) - val_size
        train_dataset, _ = random_split(
            sample_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        return DataLoader(
            train_dataset,
            batch_size=self.sample_batch_size,
            shuffle=self.shuffle,
            num_workers=self.sample_num_workers,
        )
