"""
ITER Magnetics Diagnostic Analysis Package

High-level package for anomaly detection and analysis in tokamak magnetics diagnostics
using advanced machine learning techniques including MSCRED, VAE, and SCINet architectures.

This package provides comprehensive tools for:
- Multi-scale anomaly detection with MSCRED
- Variational autoencoder-based outlier detection
- Time series prediction with SCINet
- Universal machine learning utilities
- Data downloading and preprocessing

Main Modules:
    - project_mscred: Multi-Scale Convolutional Recurrent Encoder-Decoder for anomaly detection
    - project_vae: Iterative β-VAE for outlier detection and latent space analysis
    - project_scinet: Science Network for time series prediction and latent extraction
    - ml_tools: Universal machine learning utilities and tools
    - data_downloading: MAST database integration and data acquisition

For detailed information and usage examples, see the documentation in each module's README.md file.

For detailed documentation about the project structure, see the README.md file in the repository root.
"""

__version__ = "0.1.0"
__author__ = "Louis Brusset"
__email__ = "louis.brusset@etu.minesparis.psl.eu"

# Core ML Tools - Most Important Utilities
from .ml_tools import (
    # Device management and reproducibility
    print_torch_info,
    select_torch_device,
    seed_everything,
    # Preprocessing utilities
    normalize_batch,
    denormalize_batch,
    # Projection and visualization
    project_tsne,
    project_umap,
    plot_projection,
    # Model inspection
    print_model_parameters,
)

# MSCRED - Multi-Scale Anomaly Detection
from .project_mscred import (
    # Core architectures
    MSCRED,
    ConvLSTM,
    # Configuration
    config as mscred_config,
    # Dataset utilities
    TimeSeriesDataset,
    create_data_loaders,
    # Data processing
    select_data_channels,
    build_windows,
    # Signature matrix generation
    create_signature_matrix,
    # Evaluation and detection
    load_model as load_mscred_model,
    find_anomaly_threshold,
    detect_anomalies_all,
    compute_residuals,
)

# VAE - Iterative Variational Autoencoder
from .project_vae import (
    # Core architectures
    LSTMBetaVAE,
    LengthAwareLSTMEncoder,
    LengthAwareLSTMDecoder,
    # Configuration
    config as vae_config,
    # Dataset utilities
    MultivariateTimeSerieDataset,
    create_datasets,
    # Training pipeline
    train_iterative_vae_pipeline,
    train_one_vae,
    # Outlier detection
    find_threshold_kde,
    detect_outliers_kde,
    find_cluster_and_classify,
    # Visualization
    plot_history,
    plot_projected_latent_space,
    plot_random_reconstructions,
)

# SCINet - Science Network for Time Series
from .project_scinet import (
    # Core architectures
    SciNetEncoder,
    QuestionDecoder,
    PendulumNet,
    # Configuration
    config as scinet_config,
    # Dataset utilities
    build_dataset,
    PendulumDataset,
    # Prediction and evaluation
    make_one_prediction,
    make_timeserie_prediction,
    plot_one_prediction,
    # Latent space analysis
    get_latent_activations,
    plot_3d_latent_activations,
)

# Submodule imports for advanced usage
from . import ml_tools
from . import project_mscred
from . import project_vae
from . import project_scinet
from . import data_downloading

# Public API - Most Important Components
__all__ = [
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    # Submodules
    "ml_tools",
    "project_mscred",
    "project_vae",
    "project_scinet",
    "data_downloading",
    # ML Tools - Core Utilities
    "print_torch_info",
    "select_torch_device",
    "seed_everything",
    "normalize_batch",
    "denormalize_batch",
    "project_tsne",
    "project_umap",
    "plot_projection",
    "print_model_parameters",
    # MSCRED - Anomaly Detection
    "MSCRED",
    "ConvLSTM",
    "mscred_config",
    "TimeSeriesDataset",
    "create_data_loaders",
    "select_data_channels",
    "build_windows",
    "create_signature_matrix",
    "load_mscred_model",
    "find_anomaly_threshold",
    "detect_anomalies_all",
    "compute_residuals",
    # VAE - Outlier Detection
    "LSTMBetaVAE",
    "LengthAwareLSTMEncoder",
    "LengthAwareLSTMDecoder",
    "vae_config",
    "MultivariateTimeSerieDataset",
    "create_datasets",
    "train_iterative_vae_pipeline",
    "train_one_vae",
    "find_threshold_kde",
    "detect_outliers_kde",
    "find_cluster_and_classify",
    "plot_history",
    "plot_projected_latent_space",
    "plot_random_reconstructions",
    # SCINet - Time Series Prediction
    "SciNetEncoder",
    "QuestionDecoder",
    "PendulumNet",
    "scinet_config",
    "build_dataset",
    "PendulumDataset",
    "make_one_prediction",
    "make_timeserie_prediction",
    "plot_one_prediction",
    "get_latent_activations",
    "plot_3d_latent_activations",
]
