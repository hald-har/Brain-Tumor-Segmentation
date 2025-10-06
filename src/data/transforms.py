from torchvision import transforms as T


def get_train_transforms(image_size=512):
    transforms = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )
    return transforms


def get_val_transforms(image_size=512):
    transforms = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]
    )
    return transforms
