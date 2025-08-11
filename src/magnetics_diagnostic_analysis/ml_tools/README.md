# ML Tools Module

This module contains machine learning utilities and tools for training deep learning models on magnetics diagnostic data.

## Table of Contents

1. [Files](#-files)
2. [Classes and Functions](#-classes-and-functions)
   - [`EarlyStopping`](#earlystopping)
     - [Constructor](#constructor)
     - [Methods](#methods)
       - [`check_stop`](#check_stopcurrent_loss-float-model---bool)
       - [`restore_best_weights`](#restore_best_weightsmodel---none)
3. [Usage Examples](#-usage-examples)
   - [Basic Usage](#basic-usage)
   - [Advanced Configuration](#advanced-configuration)
4. [Key Features](#-key-features)
5. [Integration with MSCRED/VAE](#-integration-with-mscredvae)
6. [Customization](#-customization)
7. [Best Practices](#-best-practices)

## 📁 Files

- **`early_stopping.py`** - Early stopping implementation for training loops

## 🔧 Classes and Functions

### `EarlyStopping`
A utility class for implementing early stopping during model training to prevent overfitting.

#### Constructor
```python
EarlyStopping(min_delta: float = 0.001, patience: int = 5)
```

**Parameters:**
- `min_delta`: Minimum change in the monitored quantity to qualify as an improvement
- `patience`: Number of epochs with no improvement after which training will be stopped

#### Methods

##### `check_stop(current_loss: float, model) -> bool`
Check if training should be stopped based on the current loss.

```python
early_stop = EarlyStopping(min_delta=0.001, patience=10)

for epoch in range(max_epochs):
    # Training loop
    train_loss = train_model()
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
