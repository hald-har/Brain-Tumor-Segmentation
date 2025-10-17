"""
This module defines a custom PyTorch Dataset for image segmentation tasks.

It handles loading image-mask pairs from specified directories.
"""

import os
import glob
import yaml
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from src.data.transforms import get_train_transforms, get_val_transforms


class BaseDataset(Dataset):
    """
    A PyTorch Dataset for loading image and mask pairs from folders.

    This class is designed to handle image segmentation datasets where
    images and their corresponding masks are stored in separate directories.
    It supports common image formats and applies optional transforms.

    Args:
        images_folder (str): The path to the folder containing input images.
        masks_folder (str): The path to the folder containing segmentation masks.
        transforms (callable, optional): A function/transform to apply to the input images.
        target_transforms (callable, optional): A function/transform to apply to the target masks.
    """

    def __init__(
        self,
        images_folder="../train/images",
        masks_folder="../train/masks",
        image_size=224,
    ):

        # Validate folder existence
        if not os.path.exists(images_folder):
            raise FileNotFoundError(f"Images folder not found: {images_folder}")
        if not os.path.exists(masks_folder):
            raise FileNotFoundError(f"Masks folder not found: {masks_folder}")

        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.image_size = image_size
        self.transforms = T.Compose(
            [
                T.Resize(
                    (self.image_size, self.image_size)
                ),  # Resize images to a fixed size
                T.ToTensor(),  # Convert images to PyTorch tensors
            ]
        )
        self.target_transforms = T.Compose(
            [
                T.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
            ]
        )
        self.image_list = sorted(
            [
                f
                for f in glob.glob(str(os.path.join(images_folder, "*")))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.mask_list = sorted(
            [
                f
                for f in glob.glob(str(os.path.join(masks_folder, "*")))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        assert len(self.image_list) == len(
            self.mask_list
        ), "Number of images and masks should be the same"

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        mask = np.array(Image.open(self.mask_list[index]).convert("L"))

        # Convert mask to binary
        mask = np.where(mask > 0, 1, 0)
        mask = Image.fromarray((mask * 255).astype(np.uint8))

        image = self.transforms(image)
        mask = self.target_transforms(mask)

        return image, mask

    def __len__(self):
        return len(self.image_list)
