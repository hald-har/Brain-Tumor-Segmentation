"""
Tests for the BaseDataset class in src.data.dataset.

This module contains unit tests to ensure the BaseDataset class
correctly loads, processes, and returns image-mask pairs.
"""
from src.data.dataset import BaseDataset

# Paths to your sample images and masks
IMAGES_FOLDER = "src/sample_images/images"
MASKS_FOLDER = "src/sample_images/masks"


def test_initialization():
    """Check that dataset loads correct number of samples."""
    dataset = BaseDataset(images_folder=IMAGES_FOLDER, masks_folder=MASKS_FOLDER)
    assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"


def test_getitem():
    """Check that __getitem__ returns valid images and masks of correct size."""
    dataset = BaseDataset(images_folder=IMAGES_FOLDER, masks_folder=MASKS_FOLDER)
    image, mask = dataset[0]

    # Check sizes (adjust 512x512 to your actual sample size)
    assert image.size == (512, 512), f"Expected image size (512,512), got {image.size}"
    assert mask.size == (512, 512), f"Expected mask size (512,512), got {mask.size}"


def test_length():
    """Check __len__ returns correct length."""
    dataset = BaseDataset(images_folder=IMAGES_FOLDER, masks_folder=MASKS_FOLDER)
    assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"


