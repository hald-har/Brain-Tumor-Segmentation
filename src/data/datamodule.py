"""
Data module for PyTorch image segmentation tasks.

Defines SegmentationDataModule to handle train/val/test/sample dataloaders
with configurable transforms and dataset paths.
"""

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

    def __init__(
        self,
        images_folder="../train/images",
        masks_folder="../train/masks",
        test_images_folder="../test/images",
        test_masks_folder="../test/masks",
        num_workers=1,
        val_split=0.1,
        shuffle=True,
        image_size=224,
        batch_size=32,
    ):

        # Dataset paths and training settings
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.test_images_folder = test_images_folder
        self.test_masks_folder = test_masks_folder
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle = shuffle
        self.image_size = image_size
        self.batch_size = batch_size
        # Sample dataset settings
        self.sample_images_folder = "src/sample_images/images"
        self.sample_masks_folder = "src/sample_images/masks"
        self.sample_batch_size = 2
        self.sample_num_workers = 1

        # Full dataset and train/val split
        self.dataset = BaseDataset(
            self.images_folder,
            self.masks_folder,
            self.image_size,
        )

        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
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
        test_dataset = BaseDataset(
            self.test_images_folder, self.test_masks_folder, self.image_size
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
            self.image_size,
        )

        return DataLoader(
            sample_dataset,
            batch_size=self.sample_batch_size,
            shuffle=self.shuffle,
            num_workers=self.sample_num_workers,
        )
