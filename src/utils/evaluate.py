import torch
from tqdm import tqdm
from src.utils.loss import dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    """
    Evaluates a segmentation model on the given dataloader using Dice coefficient.

    Args:
        net (nn.Module): The segmentation model.
        dataloader (DataLoader): Validation or test dataloader.
        device (torch.device): Device to run evaluation on.
        amp (bool): Automatic mixed precision flag (unused here).

    Returns:
        float: Average Dice coefficient over the dataset.
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # Iterate over the validation set
    for idx, (image, mask) in enumerate(
        tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation Round",
            unit="Batch",
            leave=False,
        )
    ):
        # Move tensors to the target device
        image = image.to(device, dtype=torch.float32, memory_format=torch.channels_last)
        mask = mask.to(device, dtype=torch.float32, memory_format=torch.channels_last)

        # Forward pass
        mask_pred = net(image)
        mask_pred = torch.sigmoid(mask_pred)

        # Check mask range
        assert 0 <= mask_pred.min() and mask_pred.max() <= 1, \
            f"True mask indices should be in [0,1] - MIN {mask_pred.min()} | MAX {mask_pred.max()}"

        dice_score += dice_coeff(mask_pred, mask)

    # Restore train mode
    net.train()
    return dice_score / max(num_val_batches, 1)
