"""
Project SCINet Module

Science Network (SCINet) implementation for extracting latent representations from time series data.

This module provides neural network architectures, testing utilities, and visualization tools
for SCINet-based time series modeling and anomaly detection.

Main Components:
    - Configuration management
    - Dataset building utilities
    - Model architectures (SciNetEncoder, QuestionDecoder, PendulumNet)
    - Testing and evaluation functions
    - Latent space visualization tools

For usage examples and detailed documentation, see the README.md file in this directory.
"""

__version__ = "0.1.0"
__author__ = "Louis Brusset"

# Import main model classes
from .model.scinet import (
    SciNetEncoder,
    QuestionDecoder, 
    PendulumNet
)

# Import dataset utilities
from .utils.build_dataset import (
    build_dataset,
    PendulumDataset
)

# Import testing and evaluation functions
from .utils.test_scinet import (
    make_one_prediction,
    make_timeserie_prediction,
    plot_one_prediction,
    plot_timeserie_prediction
)

# Import visualization tools
from .utils.plot_latent_activations import (
    load_trained_model,
    get_one_latent_activation,
    get_latent_activations,
    plot_3d_latent_activations
)

# Import configuration
from .setting_scinet import config

__all__ = [
    # Model architectures
    "SciNetEncoder",
    "QuestionDecoder", 
    "PendulumNet",
    
    # Dataset utilities
    "build_dataset",
    "PendulumDataset",
    
    # Testing and evaluation
    "make_one_prediction",
    "make_timeserie_prediction",
    "plot_one_prediction",
    "plot_timeserie_prediction",
    
    # Visualization tools
    "load_trained_model",
    "get_one_latent_activation", 
    "get_latent_activations",
    "plot_3d_latent_activations",
    
    # Configuration
    "config"
]