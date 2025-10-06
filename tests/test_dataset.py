"""
Tests for the dataset and dataloader modules.

- Checks if BaseDataset loads the correct number of samples.
- Validates the shape of images and masks.
- Ensures the train dataloader returns batches with correct dimensions.
"""

import os
import yaml
import torch
import pytest
from PIL import Image
import numpy as np

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
NUM_SAMPLES = config["sample_dataset"]["num_samples"]
NUM_WORKERS = config["sample_dataset"]["num_workers"]
BATCH_SIZE = config["sample_dataset"]["batch_size"]
IMAGE_SIZE = config["dataset"]["image_size"]


@pytest.fixture(scope="module")
def base_dataset():
    """Fixture to provide a ready-to-use BaseDataset."""
    return BaseDataset(
        images_folder=IMAGES_FOLDER,
        masks_folder=MASKS_FOLDER,
        image_size=IMAGE_SIZE,
    )


@pytest.fixture(scope="module")
def data_module():
    """Fixture to provide a ready-to-use SegmentationDataModule."""
    return SegmentationDataModule(
        images_folder=IMAGES_FOLDER,
        masks_folder=MASKS_FOLDER,
        test_images_folder=IMAGES_FOLDER,  # using same sample set
        test_masks_folder=MASKS_FOLDER,
        num_workers=NUM_WORKERS,
        shuffle=True,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )


def test_initialization(base_dataset):
    """Check that dataset loads correct number of samples."""
    expected = NUM_SAMPLES
    assert (
        len(base_dataset) == expected
    ), f"Expected {expected} samples, got {len(base_dataset)}"


def test_getitem(base_dataset):
    """Check that __getitem__ returns valid images and masks of correct size."""
    image, mask = base_dataset[0]

    img_size = IMAGE_SIZE
    assert isinstance(image, torch.Tensor), "Image should be a tensor"
    assert isinstance(mask, torch.Tensor), "mask should be a tensor"
    assert image.shape[0] == 3, "Image should have 3 channels (C,H,W)"
    assert mask.shape[0] == 1, "Mask should have 1 channel (C,H,W)"
    assert image.shape[1:] == (
        img_size,
        img_size,
    ), f"Expected image size {(img_size, img_size)}, got {image.shape[1:]}"
    assert mask.shape[1:] == (
        img_size,
        img_size,
    ), f"Expected mask size {(img_size, img_size)}, got {mask.shape[1:]}"


def test_length(base_dataset):
    """Check __len__ returns correct length."""
    expected = NUM_SAMPLES
    assert (
        len(base_dataset) == expected
    ), f"Expected {expected} samples, got {len(base_dataset)}"


def test_train_dataloader(data_module):
    """Test that train dataloader returns batches of correct shape."""
    sample_train_loader = data_module.sample_train_dataloader()
    images, masks = next(iter(sample_train_loader))

    assert (
        images.shape[0] == BATCH_SIZE
    ), f"Expected batch size {BATCH_SIZE}, got {images.shape[0]}"
    assert images.shape[1] == 3, "Incorrect number of image channels"
    assert masks.shape[1] == 1, "Incorrect number of mask channels"
    assert (
        images.shape[2] == IMAGE_SIZE and images.shape[3] == IMAGE_SIZE
    ), "Incorrect image size"
    assert (
        masks.shape[2] == IMAGE_SIZE and masks.shape[3] == IMAGE_SIZE
    ), "Incorrect mask size"
    print("Train dataloader test passed!")
