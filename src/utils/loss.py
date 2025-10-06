import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor):
    epsilon = 1e-6  # To avoid divide by zero
    assert (
        input.size() == target.size()
    ), f"input image size {input.size()}, doesnt match with target image size{target.size()}"

    if input.dim() == 2:
        input = input.unsqueeze(0).unsqueeze(0)

    elif input.dim() == 3:  # (Batch,height,width )
        input = input.unsqueeze(0)  # → (B,1,H,W)
        target = target.unsqueeze(0)  # → (B,1,H,W)

    # Sum over channel, height, width
    sum_dim = (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)

    # Sum of sets
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(
        sets_sum == 0, inter, sets_sum
    )  # If both masks are empty (sets_sum == 0), replace it with inter.
    dice = (inter + epsilon) / (sets_sum + epsilon)

    # return average over batch
    return dice.mean()


def dice_loss(input: Tensor, target: Tensor):
    dice = dice_coeff(input, target)
    return 1 - dice  # we do 1 - dice to minimize the loss
