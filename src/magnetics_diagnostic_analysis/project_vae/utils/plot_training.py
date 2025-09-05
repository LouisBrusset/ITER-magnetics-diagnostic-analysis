import numpy as np
import torch
import matplotlib.pyplot as plt

from magnetics_diagnostic_analysis.project_vae.setting_vae import config
from magnetics_diagnostic_analysis.ml_tools.projection_2d import project_tsne, project_umap, plot_projection



def plot_history(history, save_path=None, verbose=False) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        if verbose:
            print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
    return None



def plot_density_and_threshold(density_values, threshold, save_path=None, verbose=False) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(np.sort(density_values), np.linspace(0, 1, len(density_values)), label='Density Values')
    plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Density Value')
    plt.ylabel('Cumulative Distribution')
    plt.title('KDE Density Values with Threshold')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        if verbose:
            print(f"Density and threshold plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
    return None




def plot_projected_latent_space(latent_features, clusters, outlier_mask, save_path=None, verbose=False) -> None:
    # First, reduce to 2D using TSNE
    projection = project_tsne(latent_features, seed=config.SEED)
    
    # Plot the projection with clusters
    plot_projection(
        projection=projection,
        labels=clusters,
        title="Projected Latent Space with Clusters",
        filename=save_path if save_path else "latent_space_projection.png",
        legend=True,
        verbose=verbose
    )
    return None


def plot_random_reconstructions(reconstruction_couples, n_samples=5, save_path=None, verbose=False) -> None:
    x_init = reconstruction_couples[0]
    x_recon = reconstruction_couples[1]

    total_samples = x_init.shape[0]
    n_samples = min(n_samples, total_samples)
    random_indices = np.random.choice(total_samples, n_samples, replace=False)

    plt.figure(figsize=(15, 3 * n_samples))
    for i, idx in enumerate(random_indices):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(x_init[idx], label='Original', alpha=0.7)
        plt.plot(x_recon[idx], label='Reconstructed', alpha=0.7)
        plt.title(f'Sample {idx}')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        if verbose:
            print(f"Random reconstructions plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
    return None