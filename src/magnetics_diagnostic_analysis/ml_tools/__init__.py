"""
Machine Learning Tools Module

Universal and reusable machine learning utilities designed for training and evaluating 
deep learning models on magnetics diagnostic data.

This module provides universal tools that can be used across all package projects 
(MSCRED, VAE, SCINet) with a plug-and-play philosophy for rapid experimentation.

Main Components:
    - Training callbacks (early stopping, learning rate scheduling, gradient clipping)
    - Model-specific loss functions and metrics
    - Data preprocessing utilities
    - High-dimensional data visualization tools
    - PyTorch device management
    - Reproducibility tools
    - Model inspection utilities

For usage examples and detailed documentation, see the README.md file in this directory.
"""

__version__ = "0.1.0"
__author__ = "Louis Brusset"

# Import training callbacks
from .train_callbacks import (
    EarlyStopping,
    LRScheduling,
    GradientClipping,
    DropOutScheduling,
    EMA
)

# Import metrics and loss functions
from .metrics import (
    mscred_loss_function,
    mscred_anomaly_score,
    vae_loss_function,
    vae_reconstruction_error,
    scinet_loss
)

# Import preprocessing utilities
from .preprocessing import (
    normalize_batch,
    denormalize_batch
)

# Import projection and visualization tools
from .projection_2d import (
    project_tsne,
    project_umap,
    apply_umap,
    plot_projection
)

# Import device management
from .pytorch_device_selection import (
    print_torch_info,
    select_torch_device
)

# Import reproducibility tools
from .random_seed import (
    seed_everything
)

# Import model inspection tools
from .show_model_parameters import (
    print_model_parameters
)

__all__ = [
    # Training callbacks
    "EarlyStopping",
    "LRScheduling",
    "GradientClipping",
    "DropOutScheduling",
    "EMA",
    
    # Metrics and loss functions
    "mscred_loss_function",
    "mscred_anomaly_score",
    "vae_loss_function",
    "vae_reconstruction_error",
    "scinet_loss",
    
    # Preprocessing utilities
    "normalize_batch",
    "denormalize_batch",
    
    # Projection and visualization
    "project_tsne",
    "project_umap",
    "apply_umap",
    "plot_projection",
    
    # Device management
    "print_torch_info",
    "select_torch_device",
    
    # Reproducibility tools
    "seed_everything",
    
    # Model inspection
    "print_model_parameters"
]