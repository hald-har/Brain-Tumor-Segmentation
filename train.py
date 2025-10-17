import os
from click.core import batch
from torch.mps import device_count
import yaml
import logging

import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


from src.data.datamodule import SegmentationDataModule
from src.models.unet import UNet
from src.utils.optimizer import get_optimizer
from src.utils.loss import dice_loss
from src.utils.evaluate import evaluate
from src.utils.earlystopping import EarlyStopping


def train_model(image_folder, mask_folder, checkpoint_folder, device):
    # Initialize Logging
    experiment = wandb.init(project="U-Net", config={})
    cfg = wandb.config

    LR = cfg["learning_rate"]
    EPOCHS = cfg["epochs"]
    BATCH = cfg["batch_size"]
    FEATURES = cfg["features"]
    OPTIMIZER = cfg["optimizer"]
    AMP = cfg["amp"]
    IMAGE_SIZE = cfg["image_size"]

    # class to make dataloaders ,
    datamodule = SegmentationDataModule(
        images_folder=image_folder, masks_folder=mask_folder, image_size = IMAGE_SIZE ,batch_size=BATCH
    )
    # Create dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # model
    model = UNet(features=FEATURES).to(device)
    optimizer = get_optimizer(model.parameters(), OPTIMIZER, LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5
    )  # goal: maximize Dice score
    critereon = nn.BCEWithLogitsLoss().to(device)
    grad_scaler = torch.GradScaler(enabled=AMP)

    global_step = 0
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    num_test_batches = len(test_loader)

    n_train = len(datamodule.train_dataset)  # no of images in train dataset

    sweep_id = wandb.run.sweep_id if wandb.run is not None else "default_sweep"
    checkpoint_folder = os.path.join(checkpoint_folder, str(sweep_id))
    os.makedirs(checkpoint_folder, exist_ok=True)

    Stop_Training = False
    stopper = EarlyStopping(patience = 2 , delta = 0.005) 

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{EPOCHS}]", leave=False)
        for idx, (image, mask) in enumerate(loop):
            assert image.shape[1] == model.in_channels, (
                f"Network has been defined with {model.in_channels} input channels, "
                f"but loaded images have {image.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )
            image = image.to(
                device, dtype=torch.float32, memory_format=torch.channels_last
            )
            mask = mask.to(device, dtype=torch.float32)

            with torch.autocast(device.type, enabled=AMP):
                mask_pred = model(image)
                loss = critereon(mask_pred, mask)
                loss += dice_loss(torch.sigmoid(mask_pred), mask)
                if torch.isnan(loss):
                    print(f"NaN found in loss at global_step {global_step}")


            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1

            epoch_loss += loss.item()

            experiment.log({"Epoch": epoch, "Step": global_step, "Loss": loss.item()})

            # Evaluation Round
            division_step = n_train // (3 * num_train_batches)
            if division_step > 0:
                if global_step % division_step == 0:
                    histogram = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace("/", ".")  # Wandb prefer '.' instead of '/'
                        if not (torch.isinf(value).any() | torch.isnan(value).any()):
                            histogram["Weights" + tag] = wandb.Histogram(
                                value.detach().cpu()
                            )
                        if value.grad is not None:
                            if not (
                                torch.isinf(value.grad).any()
                                | torch.isnan(value.grad).any()
                            ):
                                histogram["Gradients" + tag] = wandb.Histogram(
                                    value.grad.detach().cpu()
                                )

                    val_score = evaluate(model, val_loader, device, AMP)
                    

                    logging.info("Validation Dice Score : {}".format(val_score))

                    try:
                        prob_pred = torch.sigmoid(mask_pred)
                        binary_pred = (prob_pred > 0.5).float()

                        experiment.log(
                            {
                                "Learning Rate": optimizer.param_groups[0]["lr"],
                                "Validation Dice": val_score,
                                "Images": wandb.Image(image[0].cpu()),
                                "Masks": {
                                    "True": wandb.Image(mask[0].cpu()),
                                    "Predicted": wandb.Image(
                                        binary_pred[0].cpu() * 255
                                    ),
                                },
                                "Step": global_step,
                                "Epoch": epoch,
                                **histogram,
                            }
                        )
                    except:
                        pass

                # Save checkpoint per epoch
        checkpoint_path = os.path.join(
            checkpoint_folder, f"checkpoint_epoch{epoch}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss / len(train_loader),
            },
            checkpoint_path,
        )
        logging.info(f"Checkpoint saved at {checkpoint_path}")


        test_score = evaluate(model, test_loader , device, AMP )
        scheduler.step(test_score)
        experiment.log({
            "Test Dice": test_score
            })
        if stopper.should_stop:
            logging.info("Early stopping triggered. Stopping training.")
            break


if __name__ == "__main__":

    if not os.path.exists("configs/config.yaml"):
        raise FileNotFoundError("Config file not found at configs/config.yaml")
    with open("configs/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_PATH = config["dataset"]["images_folder"]
    MASK_PATH = config["dataset"]["masks_folder"]
    CHECKPOINT_PATH = config["dataset"]["checkpoint"]

    train_model(IMAGE_PATH, MASK_PATH, CHECKPOINT_PATH, device)
