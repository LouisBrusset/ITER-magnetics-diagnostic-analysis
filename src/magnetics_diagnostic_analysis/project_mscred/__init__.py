"""
Project MSCRED Module

Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED) implementation for anomaly detection
in magnetics diagnostic data.

This module provides neural network architectures, signature matrix generation, evaluation utilities,
and data processing tools for MSCRED-based multivariate time series anomaly detection.

Main Components:
    - Configuration management
    - Model architectures (MSCRED, CnnEncoder, CnnDecoder, Conv_LSTM, ConvLSTM)
    - Signature matrix generation and visualization
    - Dataset building and windowing utilities
    - Evaluation and anomaly detection functions
    - Data processing and channel selection tools

For usage examples and detailed documentation, see the README.md file in this directory.
"""

__version__ = "0.1.0"
__author__ = "Louis Brusset"

# Import main model classes
from .model.mscred import CnnEncoder, Conv_LSTM, CnnDecoder, MSCRED

# Import ConvLSTM components
from .model.convlstm import ConvLSTMCell, ConvLSTM

# Import signature matrix utilities
from .utils.matrix_generator import (
    generate_signature_matrix,
    plot_signature_matrices,
    create_signature_matrix_animation,
)

# Import dataset utilities
from .utils.dataloader_building import TimeSeriesDataset, create_data_loaders

# Import windowing and data processing
from .utils.window_building import select_data_channels, build_windows

# Import evaluation and anomaly detection functions
from .utils.evaluation_mscred import (
    load_model,
    find_anomaly_threshold,
    detect_anomalies_all,
    plot_anomalies_all,
    compute_residuals,
    detect_problematic_diagnostics,
)

# Import configuration
from .setting_mscred import config

__all__ = [
    # Model architectures
    "CnnEncoder",
    "Conv_LSTM",
    "CnnDecoder",
    "MSCRED",
    # ConvLSTM components
    "ConvLSTMCell",
    "ConvLSTM",
    # Signature matrix utilities
    "generate_signature_matrix",
    "plot_signature_matrices",
    "create_signature_matrix_animation",
    # Dataset utilities
    "TimeSeriesDataset",
    "create_data_loaders",
    # Data processing and windowing
    "select_data_channels",
    "build_windows",
    # Evaluation and anomaly detection
    "load_model",
    "find_anomaly_threshold",
    "detect_anomalies_all",
    "plot_anomalies_all",
    "compute_residuals",
    "detect_problematic_diagnostics",
    # Configuration
    "config",
]
