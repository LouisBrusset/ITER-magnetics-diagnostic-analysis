# src/data/__init__.py

__all__ = []

# Standard library imports
import os
import sys

# Third-party imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
# None

def configure_plots():
    """Configure les paramètres par défaut des visualisations."""
    plt.rcParams["font.family"] = "sans"
    plt.rcParams["font.size"] = 8
    sns.set_palette('muted')

configure_plots()