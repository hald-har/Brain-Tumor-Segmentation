"""
This module defines a custom PyTorch Dataset for image segmentation tasks.

It handles loading image-mask pairs from specified directories.
"""

import os
import glob
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# Load config
if not os.path.exists("configs/config.yaml"):
    raise FileNotFoundError("Config file not found at configs/config.yaml")

with open("configs/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)



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
        self, images_folder, masks_folder, transforms=None, target_transforms=None
    ):
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.image_size = config["dataset"]["image_size"]
        self.image_list = sorted(
            [
                f
                for f in glob.glob(os.path.join(images_folder, "*"))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.mask_list = sorted(
            [
                f
                for f in glob.glob(os.path.join(masks_folder, "*"))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        # Define default transforms if none are provided
        # This ensures the data is always converted to a tensor
        # Create the full dataset
        if self.transforms is None:
            self.transforms = T.Compose(
                [
                    T.Resize(
                        (self.image_size, self.image_size)
                    ),  # Resize images to a fixed size
                    T.ToTensor(),  # Convert images to PyTorch tensors
                ]
            )

        if self.target_transforms is None:
            self.target_transforms = T.Compose(
                [
                    T.Resize(
                        (self.image_size, self.image_size)
                    ),  # Resize masks to a fixed size
                    T.ToTensor(),  # Convert masks to PyTorch tensors
                ]
            )

        assert len(self.image_list) == len(
            self.mask_list
        ), "Number of images and masks should be the same"

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert("RGB")
        mask = Image.open(self.mask_list[index]).convert("L")

        if self.transforms:
            image = self.transforms(image)

        if self.target_transforms:
            mask = self.target_transforms(mask)

        return image, mask

    def __len__(self):
        return len(self.image_list)
