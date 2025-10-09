# Project MSCRED Module

Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED) implementation for anomaly detection in magnetics diagnostic data.

MSCRED is inspired by the paper: [A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data](https://arxiv.org/abs/1811.08055)

## Table of Contents

1. [Files](#files)
2. [MSCRED Overview - Architecture](#mscred-overview---architecture)
3. [Utilisation](#utilisation)
4. [Configuration Parameters](#configuration-parameters)

## Files

### Main Files
- **`__init__.py`** - Prepare importation for future utilisation
- **`setting_mscred.py`** - Configuration and settings for MSCRED
- **`train_mscred.py`** - Training and evaluation pipeline for MSCRED model

### Model Architecture
- **`model/mscred.py`** - MSCRED neural network implementation with encoder-decoder architecture
- **`model/convlstm.py`** - ConvLSTM components for temporal modeling

### Utilities
- **`utils/matrix_generator.py`** - Signature matrix generation for multivariate time series
- **`utils/dataloader_building.py`** - Custom dataset construction and preprocessing
- **`utils/window_building.py`** - Time series windowing functions
- **`utils/evaluation_mscred.py`** - Model evaluation and testing functions
- **`utils/mast_data_scraping.py`** - MAST data downloading from the API. More infomation in `src/magnetics_diagnostic_analysis/data-downloading/`
- **`utils/synthetic_anomaly_adding.py`** - Synthetic anomaly injection for testing
- **`utils/synthetic_data_creation.py`** - Synthetic data generation utilities

### Checkpoints
- **`checkpoints/`** - Saved model checkpoints and weights during training

## MSCRED Overview - Architecture

MSCRED (Multi-Scale Convolutional Recurrent Encoder-Decoder) is a deep neural network designed for multivariate time-series anomaly detection that:

- **Multi-Scale Analysis**: Captures temporal patterns at different time scales through multiple window sizes
- **Signature Matrices**: Converts multivariate time series into signature matrices that represent relationships between variables
- **CNN Encoder**: Uses convolutional layers to extract spatial features from signature matrices
- **ConvLSTM**: Employs Convolutional LSTM with an Attention mechanism to capture temporal dependencies while preserving spatial structure
- **CNN Decoder**: Reconstructs signature matrices for anomaly detection based on reconstruction error
- **Attention Mechanism**: Incorporates attention to focus on relevant temporal segments

The workflow is:
1. Convert multivariate time series into signature matrices at multiple scales
2. Use CNN encoder to extract spatial features
3. Apply ConvLSTM to model temporal dependencies
4. Reconstruct using CNN decoder
5. Detect anomalies based on reconstruction error

## Utilisation

### Data downloading

See the `src/magnetics_diagnostic_analysis/data_downloading/` module and its `README.md` file to understand the functions we use here.

To download true tokamak data, run this:
```bash
cd src/magnetics_diagnostic_analysis/project_mscred/
# If using python .env
python utils/mast_data_scraping.py
# If using uv .venv
uv run utils/mast_data_scraping.py
```

To create synthetic multivariate signals, run that:
```bash
cd src/magnetics_diagnostic_analysis/project_mscred/
# If using python .env
python utils/synthetic_data_creation.py
# If using uv .venv
uv run utils/synthetic_data_creation.py
```

### Data preprocessing, Channel selection, Anomaly adding, Window building
```bash
cd src/magnetics_diagnostic_analysis/project_mscred/
# If using python .env
python utils/window_building.py
# If using uv .venv
uv run utils/window_building.py
```

### Dataloader creatin and training
```bash
cd src/magnetics_diagnostic_analysis/project_mscred/
# If using python .env
python train_mscrd.py
# If using uv .venv
uv run train_mscrd.py
```

### Evaluation and testing
```bash
cd src/magnetics_diagnostic_analysis/project_mscred/
# If using python .env
python utils/evaluation_mscred.py
# If using uv .venv
uv run utils/evaluation_mscred.py
```

## Configuration Parameters

All parameters are stored in the file `setting_mscred.py`.

Key parameters for MSCRED:
- **STEADY_STATE**: Whether the data downloading select only the steady state in the plasma or else the full shot
- **DATA_SHAPE**: Shape of the signature matrices (channels, height, width)
- **WINDOW_SIZES**: Multiple window sizes for multi-scale analysis, e.g.: [10, 30, 60]
- **GAP_TIME**: Step size to calculate the next window for temporal modeling
- **FIRST_LEARNING_RATE**: Initial learning rate for stable training
