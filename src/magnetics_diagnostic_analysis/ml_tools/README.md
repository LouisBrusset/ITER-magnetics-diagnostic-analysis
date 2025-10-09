# ML Tools Module

This module contains universal and reusable machine learning utilities designed for training and evaluating deep learning models on magnetics diagnostic data.

## Table of Contents

1. [Presentation and Universality Philosophy](#presentation-and-universality-philosophy)
2. [File Enumeration and Purpose](#file-enumeration-and-purpose)
3. [Functions by File](#functions-by-file)
   - [train_callbacks.py](#-train_callbackspy)
   - [metrics.py](#-metricspy)
   - [preprocessing.py](#-preprocessingpy)
   - [projection_2d.py](#-projection_2dpy)
   - [pytorch_device_selection.py](#-pytorch_device_selectionpy)
   - [random_seed.py](#-random_seedpy)
   - [show_model_parameters.py](#-show_model_parameterspy)

## Presentation and Universality Philosophy

This module was designed with a philosophy of **universality** and **reusability**. The objective is to provide generic tools that can be used:

- **Across Projects**: Compatible with all package projects (MSCRED, VAE, SCINet)
- **Framework Agnostic**: Primarily PyTorch-oriented but extensible
- **Model Independent**: Generic functions not tied to specific architectures
- **Research Ready**: Essential tools for experimentation and rapid prototyping

These utilities are designed to be **plug-and-play**: you import what you need without complex dependencies.

## File Enumeration and Purpose

### 📋 **`train_callbacks.py`**
Training callbacks to control and optimize the learning process
- Early stopping, learning rate scheduling, gradient clipping
- Classes to automate training best practices

### 📊 **`metrics.py`** 
Loss functions and evaluation metrics specific to models
- Loss functions for MSCRED, VAE, SCINet
- Anomaly scores and reconstruction metrics

### 🔧 **`preprocessing.py`**
Data preprocessing utilities
- Batch normalization/denormalization
- Standard transformations for data preparation

### 📍 **`projection_2d.py`**
Projection methods and visualization for high-dimensional spaces
- t-SNE, UMAP for visualizing latent spaces
- Plotting utilities for exploratory analysis

### 💻 **`pytorch_device_selection.py`**
Intelligent PyTorch device management (CPU/GPU/MPS)
- Automatic selection of best available device
- PyTorch system information

### 🎲 **`random_seed.py`**
Experiment reproducibility management
- Universal seeding for reproducible results
- Compatible with numpy, torch, random

### 🔍 **`show_model_parameters.py`**
Model inspection and debugging
- Detailed display of model parameters
- Architecture diagnostic tools

## Functions by File

### 📋 `train_callbacks.py`

#### `class EarlyStopping`
Early training termination to prevent overfitting
- **`__init__(min_delta, patience)`**: Threshold configuration
- **`check_stop(current_loss, model)`**: Checks if training should stop
- **`restore_best_weights(model)`**: Restores best weights

#### `class LRScheduling` 
Learning rate scheduling during training
- **`__init__(optimizer, factor, patience)`**: Scheduler configuration
- **`step(metric)`**: Updates learning rate

#### `class GradientClipping`
Gradient clipping to stabilize training
- **`__init__(max_norm)`**: Defines maximum norm
- **`clip(model)`**: Applies clipping

#### `class DropOutScheduling`
Dynamic dropout scheduling
- **`__init__(model, initial_rate, decay)`**: Dropout configuration
- **`update_dropout(epoch)`**: Updates dropout rates

#### `class EMA` 
Exponential Moving Average of model weights
- **`__init__(model, decay)`**: Initializes EMA
- **`update(model)`**: Updates moving averages

### 📊 `metrics.py`

#### `mscred_loss_function(reconstructed, original)`
MSCRED-specific loss function based on reconstruction error

#### `mscred_anomaly_score(reconstructed_valid, original)`
Calculates anomaly scores for MSCRED by comparing reconstructions and originals

#### `vae_loss_function(reconstructed, original, mu, logvar, beta)`
Complete VAE loss function: reconstruction + KL divergence with β coefficient

#### `vae_reconstruction_error(reconstructed, original, lengths)`
Reconstruction error for VAE with variable length handling

#### `scinet_loss(prediction, target, mu, logvar, kld_beta)`
Loss function for SCINet combining prediction and KL regularization

### 🔧 `preprocessing.py`

#### `normalize_batch(batch)`
Normalizes a data batch between 0 and 1
- **Input**: Tensor of shape (batch_size, ...)
- **Output**: Normalized tensor + min/max values for denormalization

#### `denormalize_batch(normalized_batch, min_vals, max_vals)`
Inverse of normalization to recover original values

### 📍 `projection_2d.py`

#### `project_tsne(embedding, perplexity, n_iter, random_state)`
t-SNE projection for visualizing high-dimensional embeddings
- **embedding**: Data to project (n_samples, n_features)
- **Returns**: 2D projection + t-SNE model

#### `project_umap(embedding, n_neighbors, min_dist, random_state)`
UMAP projection better suited for large data
- Configurable parameters to control local/global structure

#### `apply_umap(model, new_embedding)`
Applies a pre-trained UMAP model to new data

#### `plot_projection(projection, labels, title, save_path)`
Visualizes 2D projections with label-based coloring

### 💻 `pytorch_device_selection.py`

#### `print_torch_info()`
Displays PyTorch system information: version, CUDA, available devices

#### `select_torch_device(temporal_dim)`
Intelligent selection of optimal device
- **temporal_dim**: "sequential" or "parallel" to optimize based on model type (time aware models must be trained on one unique CUDA gpu)
- **Returns**: Optimal torch.device (cuda/mps/cpu)

### 🎲 `random_seed.py`

#### `seed_everything(seed)`
Universal seeding for complete reproducibility
- Configures numpy, torch, random, and torch.backends.cudnn
- **seed**: Seed for reproducibility (default: 42)

### 🔍 `show_model_parameters.py`

#### `print_model_parameters(model, model_name)`
Detailed display of model architecture and parameters
- Total parameters, trainable parameters
- Layer-by-layer breakdown for debugging

