import torch


def get_optimizer(parameters, optimizer, lr):
    if optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    return optimizer
