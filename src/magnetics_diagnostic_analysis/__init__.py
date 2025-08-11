"""
ITER Magnetics Diagnostic Analysis Package

A package for finding faulty signals in diagnostics using self-supervised learning techniques.
"""

# Import main sub-packages
from . import data_loading
from . import data_analysis
from . import ml_tools
from . import project_mscred
from . import project_vae

# Package metadata
__version__ = "0.1.0"
__author__ = "Louis Brusset"
__email__ = "louis.brusset@etu.minesparis.psl.eu"

# Expose main functionality at package level
from .data_loading import (
    shot_list,
    load_data,
    build_level_2_data_per_shot,
    build_level_2_data_all_shots
)

from .ml_tools import EarlyStopping

__all__ = [
    "data_loading",
    "data_analysis", 
    "ml_tools",
    "project_mscred",
    "project_vae",
    "shot_list",
    "load_data",
    "build_level_2_data_per_shot", 
    "build_level_2_data_all_shots",
    "EarlyStopping"
]