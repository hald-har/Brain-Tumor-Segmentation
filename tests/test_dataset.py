"""
Tests for the dataset and dataloader modules.

- Checks if BaseDataset loads the correct number of samples.
- Validates the shape of images and masks.
- Ensures the train dataloader returns batches with correct dimensions.
"""

import os
import yaml

from src.data.dataset import BaseDataset
from src.data.datamodule import SegmentationDataModule

# Load config
if not os.path.exists("configs/config.yaml"):
    raise FileNotFoundError("Config file not found at configs/config.yaml")

with open("configs/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# Sample dataset paths from config
IMAGES_FOLDER = config["sample_dataset"]["images_folder"]
MASKS_FOLDER = config["sample_dataset"]["masks_folder"]


def test_initialization():
    """Check that dataset loads correct number of samples."""
    dataset = BaseDataset(images_folder=IMAGES_FOLDER, masks_folder=MASKS_FOLDER)
    expected = config["sample_dataset"]["num_samples"]
    assert len(dataset) == expected, (
        f"Expected {expected} samples, got {len(dataset)}"
    )


def test_getitem():
    """Check that __getitem__ returns valid images and masks of correct size."""
    dataset = BaseDataset(images_folder=IMAGES_FOLDER, masks_folder=MASKS_FOLDER)
    image, mask = dataset[0]

    img_size = config["dataset"]["image_size"]
    assert image.shape[1:] == (img_size, img_size), (
        f"Expected image size {(img_size, img_size)}, got {image.shape[1:]}"
    )
    assert mask.shape[1:] == (img_size, img_size), (
        f"Expected mask size {(img_size, img_size)}, got {mask.shape[1:]}"
    )


def test_length():
    """Check __len__ returns correct length."""
    dataset = BaseDataset(images_folder=IMAGES_FOLDER, masks_folder=MASKS_FOLDER)
    expected = config["sample_dataset"]["num_samples"]
    assert len(dataset) == expected, (
        f"Expected {expected} samples, got {len(dataset)}"
    )


def test_train_dataloader():
    """Test that train dataloader returns batches of correct shape."""
    print("Testing Train DataLoader")
    data_module = SegmentationDataModule(config_path="configs/config.yaml")
    sample_train_loader = data_module.sample_train_dataloader()

    images, masks = next(iter(sample_train_loader))

    batch_size = config["sample_dataset"]["batch_size"]
    img_size = config["dataset"]["image_size"]

    assert images.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {images.shape[0]}"
    )
    assert masks.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {masks.shape[0]}"
    )
    assert images.shape[1] == 3, "Incorrect number of image channels"
    assert masks.shape[1] == 1, "Incorrect number of mask channels"
    assert images.shape[2] == img_size and images.shape[3] == img_size, "Incorrect image size"
    assert masks.shape[2] == img_size and masks.shape[3] == img_size, "Incorrect mask size"
    print("Train dataloader test passed!")
