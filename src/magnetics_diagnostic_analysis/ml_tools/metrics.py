import torch


def mscred_loss_function(reconstructed: torch.Tensor, original: torch.Tensor):
    """
    Loss function for measuring reconstruction error.

    Nota bene:
        - We sum over the spatial dimensions (window_size, height and width).
        - We also take the mean over the batch dimension (corresponding to time here).
    """
    squared_error = (reconstructed - original) ** 2
    loss = torch.mean(torch.sum(squared_error, dim=(1, 2, 3)))
    return loss