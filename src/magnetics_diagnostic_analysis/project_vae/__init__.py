"""
Project Iterative beta-VAE Module

Variational Autoencoder (VAE) implementation for anomaly detection in magnetics diagnostic data.
The idea is to iterate with VAE reconstruction training, then outliers elimination, and finally 
clusterization of the obtained latent space.

This module provides neural network architectures, dataset utilities, and visualization tools
for VAE-based anomaly detection and latent space analysis.

Main Components:
    - Configuration management
    - Dataset building utilities  
    - Model architectures (LengthAwareLSTMEncoder, LengthAwareLSTMDecoder, LSTMBetaVAE)
    - Training visualization and plotting functions
    - Iterative VAE training pipeline

For usage examples and detailed documentation, see the README.md file in this directory.
"""

__version__ = "0.1.0"
__author__ = "Louis Brusset"

# Import main model classes
from .model.lstm_vae import (
    LengthAwareLSTMEncoder,
    LengthAwareLSTMDecoder,
    LSTMBetaVAE
)

# Import dataset utilities
from .utils.dataset_building import (
    find_seq_length,
    MultivariateTimeSerieDataset,
    OneVariableTimeSerieDataset,
    create_datasets
)

# Import training utilities
from .train_vae import (
    pad_sequences_smartly,
    train_one_vae,
    train_final_vae,
    train_iterative_vae_pipeline,
    find_threshold_kde,
    detect_outliers_kde,
    find_cluster_and_classify
)


# Import visualization and plotting tools
from .utils.plot_training import (
    plot_history,
    plot_density_and_threshold,
    plot_projected_latent_space,
    plot_random_reconstructions
)

# Import configuration
from .setting_vae import config

__all__ = [
    # Model architectures
    "LengthAwareLSTMEncoder",
    "LengthAwareLSTMDecoder", 
    "LSTMBetaVAE",
    
    # Dataset utilities
    "find_seq_length",
    "MultivariateTimeSerieDataset",
    "OneVariableTimeSerieDataset",
    "create_datasets",

    # Training functions
    "pad_sequences_smartly",
    "train_one_vae",
    "train_final_vae",
    "train_iterative_vae_pipeline",
    "find_threshold_kde",
    "detect_outliers_kde",
    "find_cluster_and_classify",
    
    # Visualization and plotting tools
    "plot_history",
    "plot_density_and_threshold",
    "plot_projected_latent_space",
    "plot_random_reconstructions",
    
    # Configuration
    "config"
]