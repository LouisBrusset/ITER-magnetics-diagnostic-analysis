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


def mscred_anomaly_score(
    reconstructed_valid: torch.Tensor, original: torch.Tensor
) -> np.ndarray:
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


def vae_loss_function(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    z_mean: torch.Tensor,
    z_logvar: torch.Tensor,
    lengths: torch.Tensor,
    beta: float = 1.0,
    fft_weight: float = 0.2,
) -> tuple[torch.Tensor]:

    _, seq_length, _ = x.shape

    mask = (
        torch.arange(seq_length, device=x_recon.device)[None, :] < lengths[:, None]
    )  # shape [batch_size, max_length]
    mask = mask.unsqueeze(-1).float()  # shape [batch_size, max_length, 1]

    MSE = torch.nn.functional.mse_loss(x_recon, x, reduction="none")
    MSE = (MSE * mask).sum(dim=(1, 2))  # Mask application
    num_valid_steps = mask.sum(dim=(1, 2))  # Normalizing factor
    MSE = torch.where(num_valid_steps > 0, MSE / num_valid_steps, torch.zeros_like(MSE))

    KLD = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)

    x_fft = torch.fft.fft(x, dim=1)
    recon_fft = torch.fft.fft(x_recon, dim=1)
    x_mag, x_phase = torch.abs(x_fft), torch.angle(x_fft)
    recon_mag, recon_phase = torch.abs(recon_fft), torch.angle(recon_fft)
    mag_loss = torch.nn.functional.l1_loss(recon_mag, x_mag, reduction="mean")
    phase_loss = torch.nn.functional.l1_loss(recon_phase, x_phase, reduction="mean")
    FFT = mag_loss + phase_loss

    MSE = torch.mean(MSE)
    KLD = torch.mean(KLD)
    TOTAL = MSE + beta * KLD + fft_weight * FFT

    return TOTAL, MSE, KLD


def vae_reconstruction_error(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor]:

    _, seq_length, _ = x.shape

    mask = (
        torch.arange(seq_length, device=x_recon.device)[None, :] < lengths[:, None]
    )  # shape [batch_size, max_length]
    mask = mask.unsqueeze(-1).float()  # shape [batch_size, max_length, 1]

    mse = torch.nn.functional.mse_loss(x_recon, x, reduction="none")
    mse = (mse * mask).sum(dim=(1, 2))  # Mask application
    num_valid_steps = mask.sum(dim=(1, 2))  # Normalizing factor
    mse = torch.where(num_valid_steps > 0, mse / num_valid_steps, torch.zeros_like(mse))

    return mse


def scinet_loss(
    possible_answer: torch.Tensor,
    a_corr: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.001,
) -> torch.Tensor:
    # prediction_loss = nn.MSELoss(reduction='none')(possible_answer, a_corr)
    # kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).unsqueeze(-1)
    # total_loss = prediction_loss + beta * kld_loss
    # return torch.mean(total_loss)

    recon_loss = torch.nn.MSELoss()(possible_answer.squeeze(), a_corr.squeeze())
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss + beta * kld_loss, kld_loss, recon_loss
