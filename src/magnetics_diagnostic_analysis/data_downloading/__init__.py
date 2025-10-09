"""
Data Downloading Module

Comprehensive utilities for downloading, loading, and preprocessing experimental data from 
the MAST (Mega Amp Spherical Tokamak) experiment database.

This module serves as the data foundation for all machine learning projects in the package,
providing robust data pipeline from raw database to ML-ready datasets with quality control
and physics-informed filtering.

Main Components:
    - MAST database integration and data acquisition
    - Dataset building with train/test splitting
    - Steady state detection for tokamak operations
    - Data cleaning and preprocessing utilities
    - Quality control and validation tools
    - Retry mechanisms for robust data downloading

For usage examples and detailed documentation, see the README.md file in this directory.
"""

__version__ = "0.1.0"
__author__ = "Louis Brusset"

# Import core data downloading functions
from .data_downloading import (
    shot_list,
    to_dask,
    retry_to_dask,
    build_level_2_data_all_shots,
    load_data
)

# Import steady state filtering utilities
from .steady_state_filtering import (
    ip_filter
)

# Import data washing and cleaning functions
from .data_washing import (
    print_dataset_info,
    filter_xr_dataset_channels,
    impute_to_zero,
    clean_data
)

__all__ = [
    # Core data downloading functions
    "shot_list",
    "to_dask",
    "retry_to_dask",
    "build_level_2_data_all_shots",
    "load_data",
    
    # Steady state filtering
    "ip_filter",
    
    # Data washing and cleaning
    "print_dataset_info",
    "filter_xr_dataset_channels",
    "impute_to_zero",
    "clean_data"
]

