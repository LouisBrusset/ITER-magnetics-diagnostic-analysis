# Experiments Notebooks

This directory contains comprehensive experimental notebooks for testing and evaluating machine learning models on tokamak magnetics diagnostic data. Each notebook focuses on specific aspects of anomaly detection, time series prediction, and signal analysis using advanced neural network architectures.

## Table of Contents

1. [Overview](#overview)
2. [Notebook Descriptions](#notebook-descriptions)
3. [Data Dependencies](#data-dependencies)
4. [Notes](#notes)

## Overview

The notebooks from `experiments` directory implement and test four main machine learning approaches:

- **MSCRED**: Multi-Scale Convolutional Recurrent Encoder-Decoder for anomaly detection
- **VAE**: β-Variational Autoencoder for outlier detection and classification
- **SCINet**: Science aware Network for time series forecasting and physical parameter recovery
- **Spectral Analysis**: Frequency domain analysis of magnetic sensor signatures

## Notebook Descriptions

### 1. `efit++_analysis.ipynb`

**Subject**: Analysis and evaluation of EFIT++ equilibrium reconstruction data for plasma physics research.

**Purpose**: Try to find a way to label our magnetics probes data through the lifetime of the tokamak machine.

### 2. `spectral_signature.ipynb`

**Subject**: Frequency domain analysis of magnetic sensor signatures to identify characteristic spectral patterns in tokamak diagnostics.

**Purpose**: Characterizes frequency signatures of different plasma states and identifies spectral markers for labelling the data. The goal is to be able to recognize any diagnostic signal thanks to its frequency signature.

### 3. `mscred_test_synthetic_data.ipynb`

**Subject**: Testing MSCRED (Multi-Scale Convolutional Recurrent Encoder-Decoder) architecture on synthetic anomaly data for validation purposes.

**Purpose**: Validates MSCRED model performance on controlled synthetic data with known anomalies before applying to real tokamak data.

### 4. `vae_test_loop_pipline.ipynb`

**Subject**: Comprehensive β-VAE (β-Variational Autoencoder) pipeline for outlier detection and classification in tokamak diagnostic data.

**Purpose**: Implements robust outlier detection system capable of handling variable-length tokamak shot data with automatic β-parameter optimization.

### 5. `scinet_on_synthetic_pendulum.ipynb`

**Subject**: Testing SCINet model on synthetic damped pendulum data to validate time series forecasting capabilities.

**Purpose**: Validates SCINet forecasting accuracy on well-understood physical systems before application to complex plasma dynamics.

### 6. `scinet_on_abnormal_data.ipynb`

**Subject**: Application of SCINet (Sample Convolution and Interaction Network) for detecting and predicting abnormal patterns in tokamak diagnostic data.

**Purpose**: Creation of abnormal signals and try to see if SCINet can detect and predict those abnormal patterns. Or if it can generalize to unseen anomalies.



## Data Dependencies

### Required Data Files
- `data/preprocessed/mscred/data_magnetics_mscred_cleaned.nc`: Cleaned magnetic diagnostic data
- `data/preprocessed/mscred/signature_matrices.npy`: Pre-computed signature matrices
- `data/preprocessed/vae/dataset_magnetics_vae_*.pt`: VAE training/test datasets
- `data/synthetic/scinet/pendulum_scinet_*.pt`: Synthetic pendulum datasets

### External Data Sources
- **MAST Zarr Store**: `https://s3.echo.stfc.ac.uk/mast/level{level}/shots/{shot}.zarr`
- **Magnetic Diagnostics**: Multi-channel time series from tokamak sensors

### Configuration Files
- `magnetics_diagnostic_analysis.project_vae.setting_vae.config`: VAE hyperparameters
- Model parameter files in `results/model_params/`: Trained model weights

## Notes

- All notebooks support GPU acceleration via PyTorch CUDA
- Variable sequence lengths are handled through padding and masking
- Experiments include both synthetic validation and real tokamak data testing
- Results are automatically saved to `results/figures/` and `results/model_params/`
- Integration with MAST database enables cloud-based data access for large-scale experiments