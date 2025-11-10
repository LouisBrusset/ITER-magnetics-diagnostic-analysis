import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from magnetics_diagnostic_analysis.project_scinet.setting_scinet import config
from magnetics_diagnostic_analysis.project_scinet.utils.data_creation_pendulum import (
    data_synthetic_pendulum,
)
from magnetics_diagnostic_analysis.project_scinet.model.scinet import PendulumNet


def load_trained_model(
    model_path: str, device: torch.device = torch.device("cpu")
) -> PendulumNet:
    """
    Load a trained PendulumNet model from a file.

    Args:
        model_path (str): Path to the model file.
        device (torch.device, optional): Device to load the model onto. Defaults to CPU.

    Returns:
        PendulumNet: The loaded model.
    """
    torch.cuda.empty_cache()
    model = PendulumNet(
        input_size=config.M_INPUT_SIZE,
        enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
        latent_size=config.M_LATENT_SIZE,
        question_size=config.M_QUESTION_SIZE,
        dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
        output_size=config.M_OUTPUT_SIZE,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_one_latent_activation(
    model: nn.Module, observation: np.array, device: torch.device = torch.device("cpu")
) -> np.array:
    """
    Get the latent activation for a single observation.

    Args:
        model (nn.Module): The trained model.
        observation (np.array): The input observation.
        device (torch.device, optional): The device to run the model on. Defaults to CPU.

    Returns:
        np.array: The latent activation.
    """
    torch.cuda.empty_cache()
    model.to(device).eval()
    observation_tensor = (
        torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
    )
    latent_activations = []
    with torch.no_grad():
        mean, _ = model.encoder(observation_tensor)
        latent_activations.append(mean.cpu().numpy())
    return np.array(latent_activations)


def get_latent_activations(
    model: nn.Module,
    kapa_range: tuple,
    b_range: tuple,
    pixel_by_line: int = 50,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.array]:
    """
    Get the latent activations for a grid of (kapa, b) values.

    Args:
        model (nn.Module): The trained model.
        kapa_range (tuple): Range of kapa values (min, max).
        b_range (tuple): Range of b values (min, max).
        pixel_by_line (int, optional): Number of points per axis. Defaults to 50.
        device (torch.device, optional): The device to run the model on. Defaults to CPU

    Returns:
        3-tuple containing:
            np.array: The grid of kapa values
            np.array: The grid of b values
            np.array: The latent activations for the grid of (kapa, b) values
    """
    # Create a grid of (kapa, b) values
    kapa_values = np.linspace(*kapa_range, pixel_by_line)
    b_values = np.linspace(*b_range, pixel_by_line)
    kapa_grid, b_grid = np.meshgrid(kapa_values, b_values)

    # Loop over the grid and get latent activations
    latent_activations = []
    for kapa in kapa_values:
        for b in b_values:
            # Generate observation
            observation = data_synthetic_pendulum(
                kapa, b, timesteps=config.TIMESTEPS, maxtime=config.MAXTIME
            )
            # Find latent activation with encoder only
            latent_act = get_one_latent_activation(model, observation, device=device)
            latent_activations.append(latent_act)
    latents = np.array(latent_activations).reshape(pixel_by_line, pixel_by_line, -1)

    return kapa_grid, b_grid, latents


def plot_3d_latent_activations(
    kapa_grid: np.array,
    b_grid: np.array,
    latent_activations: np.array,
    save_path: str,
    shared_scale: bool = False,
) -> None:
    """ "
    Plot the latent activations in 3D for each latent dimension.

    Args:
        kapa_grid (np.array): The grid of kapa values.
        b_grid (np.array): The grid of b values.
        latent_activations (np.array): The latent activations for the grid of (kapa, b) values.
        save_path (str): Path to save the plot.
        shared_scale (bool, optional): Whether to use a shared z-axis scale for all plots. Defaults to False.

    Returns:
        None
        Figures are saved to the specified path.
    """
    latent_dim = latent_activations.shape[2]
    fig = plt.figure(figsize=(6 * latent_dim, 8))

    if shared_scale:
        z_min = np.min(latent_activations)
        z_max = np.max(latent_activations)
    else:
        z_min = z_max = None

    # Plot each latent dimension
    for i in range(latent_dim):
        ax = fig.add_subplot(1, latent_dim, i + 1, projection="3d")
        surf = ax.plot_surface(
            kapa_grid,
            b_grid,
            latent_activations[:, :, i],
            alpha=0.8,
            cmap="viridis",
            label=f"Latent {i+1}",
        )

        if shared_scale:
            ax.set_zlim(z_min, z_max)
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel(r"$b$")
        ax.set_title(f"Latent Dimension {i+1}")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return None


if __name__ == "__main__":

    device = config.DEVICE
    path = config.DIR_MODEL_PARAMS / f"{config.BEST_MODEL_NAME}.pth"
    pendulum_net = load_trained_model(path, device=device)

    kapa_range = config.KAPA_RANGE
    b_range = config.B_RANGE

    kapa_grid, b_grid, latent_activations = get_latent_activations(
        pendulum_net, kapa_range, b_range, device=device
    )
    print("Latent activations computed.")

    path = config.DIR_FIGURES / "latent_activations_3d_2.png"
    plot_3d_latent_activations(
        kapa_grid, b_grid, latent_activations, save_path=path, shared_scale=True
    )
    print(f"Latent activations plotted. Saved at: {path}")
