# ML Tools Module

This module contains machine learning utilities and tools for training deep learning models on magnetics diagnostic data.

## Table of Contents

1. [Files](#-files)
2. [Core Components](#-core-components)
   - [Training Callbacks](#training-callbacks)
   - [Metrics](#metrics)
   - [Preprocessing](#preprocessing)
   - [Device Selection](#device-selection)
   - [Other Utilities](#other-utilities)
3. [Usage Examples](#-usage-examples)
4. [Key Features](#-key-features)
5. [Best Practices](#-best-practices)

## 📁 Files

- **`train_callbacks.py`** - Training callbacks including early stopping
- **`metrics.py`** - Evaluation metrics for model performance
- **`preprocessing.py`** - Data preprocessing utilities
- **`projection_2d.py`** - 2D projection methods for visualization
- **`pytorch_device_selection.py`** - PyTorch device selection utilities
- **`random_seed.py`** - Random seed management for reproducibility
- **`show_model_parameters.py`** - Model parameter inspection tools

## 🔧 Core Components

### Training Callbacks
Training utilities including early stopping and other callback functions.

```python
from magnetics_diagnostic_analysis.ml_tools import train_callbacks

# Example usage for training callbacks
callback = train_callbacks.EarlyStopping(patience=10, min_delta=0.001)
```

### Metrics
Evaluation metrics for assessing model performance.

```python
from magnetics_diagnostic_analysis.ml_tools import metrics

# Calculate evaluation metrics
score = metrics.calculate_metric(predictions, targets)
```

### Preprocessing
Data preprocessing functions for machine learning pipelines.

```python
from magnetics_diagnostic_analysis.ml_tools import preprocessing

# Preprocess data for ML models
processed_data = preprocessing.preprocess(raw_data)
```

### Device Selection
PyTorch device selection and management.

```python
from magnetics_diagnostic_analysis.ml_tools import pytorch_device_selection

# Select appropriate device (CPU/GPU)
device = pytorch_device_selection.get_device()
```

### Other Utilities

- **2D Projections**: Visualization utilities for high-dimensional data
- **Random Seed**: Reproducibility tools for consistent results
- **Model Parameters**: Tools for inspecting model architecture and parameters
    val_loss = validate_model()
    
    # Check if should stop
    if early_stop.check_stop(val_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break
```

##### `restore_best_weights(model) -> None`
Restore the model weights from the best epoch.

```python
# After training, restore best weights
early_stop.restore_best_weights(model)
```

## 🚀 Usage Examples

### Basic Usage
```python
from magnetics_diagnostic_analysis.ml_tools import EarlyStopping
import torch

# Initialize early stopping
early_stop = EarlyStopping(min_delta=0.001, patience=5)

# Training loop
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(100):
    # Training step
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation step
    model.eval()
    val_loss = validate(model, val_loader)
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Check early stopping
    if early_stop.check_stop(val_loss, model):
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Restore best model weights
early_stop.restore_best_weights(model)
```

### Advanced Configuration
```python
# More sensitive early stopping
sensitive_early_stop = EarlyStopping(min_delta=0.0001, patience=15)

# Less sensitive early stopping (for noisy loss curves)
robust_early_stop = EarlyStopping(min_delta=0.01, patience=3)
```

## 🎯 Key Features

- **Automatic best model saving**: Keeps track of the best model weights
- **Configurable sensitivity**: Adjust `min_delta` and `patience` for your use case
- **PyTorch compatible**: Works with any PyTorch model
- **Memory efficient**: Only stores the best model state

## 📊 Integration with MSCRED/VAE

This early stopping utility is designed to work seamlessly with the anomaly detection models:

```python
# For MSCRED training
from magnetics_diagnostic_analysis.ml_tools import EarlyStopping

early_stop = EarlyStopping(min_delta=0.001, patience=10)
# Use in MSCRED training loop

# For VAE training
early_stop_vae = EarlyStopping(min_delta=0.0005, patience=15)
# Use in VAE training loop
```

## 🔧 Customization

The early stopping implementation can be extended for more complex scenarios:

- **Multiple metrics monitoring**: Modify to track validation accuracy, F1-score, etc.
- **Learning rate scheduling**: Integrate with learning rate schedulers
- **Model checkpointing**: Save complete model checkpoints at best epochs

## 💡 Best Practices

- **Start with default values**: `min_delta=0.001, patience=5`
- **Adjust patience for dataset size**: Larger datasets may need higher patience
- **Monitor validation loss**: Use validation set, not training loss
- **Save early**: Call `restore_best_weights()` after training completes
