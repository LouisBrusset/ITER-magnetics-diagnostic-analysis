# Project SCINet Module

Sample Convolution and Interaction Network (SCINet) implementation for time series analysis and prediction on magnetics diagnostic data.

## Table of Contents

1. [Files](#-files)
2. [SCINet Overview](#-scinet-overview)
3. [Core Components](#-core-components)
4. [Usage Example](#-usage-example)
5. [Model Architecture](#-model-architecture)
6. [Training and Evaluation](#-training-and-evaluation)
7. [Configuration Parameters](#-configuration-parameters)
8. [Best Practices](#-best-practices)

## 📁 Files

### Main Files
- **`setting_scinet.py`** - Configuration and settings for SCINet
- **`train_scinet.py`** - Training pipeline for SCINet model

### Model Architecture
- **`model/scinet.py`** - SCINet neural network implementation

### Utilities
- **`utils/build_dataset.py`** - Dataset construction and preprocessing
- **`utils/data_creation_pendulum.py`** - Synthetic pendulum data generation for testing
- **`utils/plot_latent_activations.py`** - Visualization of model internal activations
- **`utils/test_scinet.py`** - Model testing and evaluation functions

### Checkpoints
- **`checkpoints/`** - Saved model checkpoints and weights

## 🔬 SCINet Overview

SCINet (Sample Convolution and Interaction Network) is a neural network designed for time series forecasting that:
- Uses sample convolution to capture temporal patterns
- Employs interaction mechanisms for feature learning
- Provides efficient time series prediction capabilities
- Supports both univariate and multivariate time series

## 🔧 Core Components

### Model Training
```python
from magnetics_diagnostic_analysis.project_scinet import train_scinet

# Train SCINet model
train_scinet.main()
```

### Configuration
```python
from magnetics_diagnostic_analysis.project_scinet import setting_scinet

# Load SCINet settings
config = setting_scinet.load_config()
```

### Dataset Building
```python
from magnetics_diagnostic_analysis.project_scinet.utils import build_dataset

# Build dataset for SCINet training
dataset = build_dataset.create_dataset(data)
```

### Testing and Evaluation
```python
from magnetics_diagnostic_analysis.project_scinet.utils import test_scinet

# Test trained SCINet model
results = test_scinet.evaluate_model(model, test_data)
```

### Visualization
```python
from magnetics_diagnostic_analysis.project_scinet.utils import plot_latent_activations

# Visualize model activations
plot_latent_activations.plot_activations(model, data)
```

## 🏗️ Model Architecture

SCINet employs a hierarchical structure with:
- **Sample Convolution**: Efficient temporal pattern extraction
- **Interaction Blocks**: Cross-feature learning mechanisms
- **Recursive Structure**: Multi-scale temporal modeling
- **Skip Connections**: Enhanced gradient flow

## 🚀 Training and Evaluation

### Training Pipeline
1. Data preprocessing and windowing
2. Model configuration setup
3. Training loop with checkpointing
4. Validation and early stopping

### Evaluation Metrics
- Time series forecasting accuracy
- Computational efficiency
- Model interpretability measures

## ⚙️ Configuration Parameters

Key parameters for SCINet training:
- **Window size**: Input sequence length
- **Prediction horizon**: Output sequence length
- **Model depth**: Number of interaction blocks
- **Learning rate**: Optimization parameters

## 🎯 Best Practices

- Use appropriate window sizes for your time series
- Monitor training stability with visualization tools
- Validate on held-out test sets
- Consider computational requirements for deployment