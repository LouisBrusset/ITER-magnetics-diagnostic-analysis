# Project MSCRED Module

Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED) implementation for anomaly detection in magnetics diagnostic data.

## Table of Contents

1. [Files](#-files)
2. [MSCRED Overview](#-mscred-overview)
3. [Core Functions](#-core-functions)
   - [`generate_signature_matrix_node`](#generate_signature_matrix_node)
   - [`generate_train_test_data`](#generate_train_test_data)
4. [Signature Matrix Generation](#-signature-matrix-generation)
5. [Usage Example](#-usage-example)
6. [Development Notebook](#-development-notebook)
7. [Anomaly Detection Strategy](#-anomaly-detection-strategy)
8. [Configuration Parameters](#-configuration-parameters)
9. [Integration with Magnetics Data](#-integration-with-magnetics-data)
10. [Best Practices](#-best-practices)

## 📁 Files

- **`data_scraping.py`** - Data preparation and scraping utilities for MSCRED
- **`matrix_generator.py`** - Signature matrix generation for multivariate time series
- **`test_grandes_lignes.ipynb`** - Main development and testing notebook

## 🔬 MSCRED Overview

MSCRED is a deep neural network designed for multivariate time-series anomaly detection that:
- Captures temporal correlations in multivariate time series
- Generates signature matrices to represent time series relationships
- Uses multi-scale convolution for feature extraction
- Employs encoder-decoder architecture for reconstruction-based anomaly detection

## 🔧 Core Functions

### `generate_signature_matrix_node()`
Generate signature matrices that capture relationships between time series variables.

```python
from magnetics_diagnostic_analysis.project_mscred import generate_signature_matrix_node

# Generate signature matrix for correlation analysis
signature_matrix = generate_signature_matrix_node()
```

### `generate_train_test_data()`
Prepare training and testing datasets in the format required by MSCRED.

```python
from magnetics_diagnostic_analysis.project_mscred import generate_train_test_data

# Generate formatted data for MSCRED training
train_data, test_data = generate_train_test_data()
```

## 📊 Signature Matrix Generation

The signature matrix is a key component that captures:

### Correlation Measures
- **Pearson correlation**: Linear relationships between variables
- **Cosine similarity**: Angular similarity between time series
- **Custom metrics**: Domain-specific relationship measures

### Implementation Details
```python
# Example signature matrix generation workflow
import numpy as np
from scipy.stats import pearsonr
from scipy import spatial

# Calculate correlations between all variable pairs
def compute_signature_matrix(data):
    n_vars = data.shape[1]
    signature_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                signature_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(data[:, i], data[:, j])
                signature_matrix[i, j] = abs(corr)
    
    return signature_matrix
```

## 🚀 Usage Example

### Basic MSCRED Workflow
```python
from magnetics_diagnostic_analysis.project_mscred import (
    generate_signature_matrix_node,
    generate_train_test_data
)

# 1. Load your magnetics data
data = load_magnetics_data()

# 2. Generate signature matrices
signature_matrices = generate_signature_matrix_node()

# 3. Prepare training data
train_data, test_data = generate_train_test_data()

# 4. Train MSCRED model (implementation in notebook)
# model = train_mscred_model(train_data, signature_matrices)

# 5. Detect anomalies
# anomalies = detect_anomalies(model, test_data)
```

## 📈 Development Notebook

The `test_grandes_lignes.ipynb` notebook contains:

### 1. Synthetic Data Generation
```python
def generate_multivariate_ts(n_variables, n_timesteps, noise_level=0.1, seed=None):
    """
    Generate realistic multivariate time series with temporal continuity.
    
    Features:
    - AR(1) processes for temporal continuity
    - Unique parameters per variable (means, volatilities, trends)
    - Sinusoidal components with different frequencies
    - Configurable noise levels
    """
```

### 2. MSCRED Architecture
- **Encoder**: Multi-scale convolutional layers
- **Decoder**: Reconstruction layers
- **Attention mechanisms**: For focusing on important features
- **Loss functions**: Reconstruction + regularization terms

### 3. Training Pipeline
- Data preprocessing and normalization
- Signature matrix computation
- Model training with early stopping
- Anomaly score computation

## 🎯 Anomaly Detection Strategy

### Normal Behavior Learning
1. **Training Phase**: Learn to reconstruct normal magnetics patterns
2. **Signature Learning**: Capture variable relationships during normal operation
3. **Multi-scale Analysis**: Different temporal scales for comprehensive analysis

### Anomaly Detection
1. **Reconstruction Error**: High error indicates anomalous behavior
2. **Signature Deviation**: Changes in variable relationships
3. **Threshold Setting**: Statistical or learned thresholds for classification

## 🔧 Configuration Parameters

### Model Architecture
- **Input dimensions**: Based on number of magnetics channels
- **Convolution scales**: Multiple scales for temporal analysis
- **Encoder depth**: Number of encoding layers
- **Latent dimension**: Compressed representation size

### Training Parameters
- **Learning rate**: Adaptive or fixed
- **Batch size**: Depends on sequence length and memory
- **Epochs**: With early stopping
- **Regularization**: L1/L2, dropout rates

## 📊 Integration with Magnetics Data

### Data Preprocessing
```python
# Typical preprocessing pipeline
def preprocess_magnetics_data(raw_data):
    # 1. Permanent state filtering
    filtered_data = apply_permanent_state_filter(raw_data)
    
    # 2. Normalization
    normalized_data = normalize_time_series(filtered_data)
    
    # 3. Sequence creation
    sequences = create_sequences(normalized_data, sequence_length=100)
    
    return sequences
```

### Variable Selection
- Focus on high-quality magnetics channels
- Include EFIT reconstructions for context
- Filter based on data completeness

## 💡 Best Practices

- **Data Quality**: Ensure good data quality before training
- **Sequence Length**: Balance between context and computational efficiency
- **Validation Strategy**: Use temporal splits (not random) for time series
- **Threshold Tuning**: Use validation set for threshold optimization
- **Interpretability**: Analyze which variables contribute most to anomaly scores
