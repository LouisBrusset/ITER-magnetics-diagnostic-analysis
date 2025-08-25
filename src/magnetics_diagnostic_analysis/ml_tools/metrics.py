import torch
import numpy as np


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



def mscred_anomaly_score(reconstructed_valid: torch.Tensor, original: torch.Tensor) -> np.ndarray:
    """
    Compute the anomaly score for the reconstructed validation set.

    Returns:
        A 1D numpy array containing the anomaly scores for each sample in the batch / the timeserie.
    """
    # Possible shapes: 
    # (batch_size, channel, height, width)
    # (batch_size, channel, height, width)
    residuals = (reconstructed_valid - original) ** 2
    anomaly_score = np.sum(residuals.cpu().numpy(), axis=(1, 2, 3))
    return anomaly_score