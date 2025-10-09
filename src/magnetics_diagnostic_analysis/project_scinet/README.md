# Project SCINet Module

Science network (SCINet) implementation for time series analysis, representation learning and prediction on magnetics diagnostic data.

Scinet is inspired of this paper: [Discover physical concepts and equations with machine learning](https://arxiv.org/abs/2412.12161)

## Table of Contents

1. [Files](#-files)
2. [SCINet Overview - Architecture](#-scinet-overview---architecture)
3. [Utilisation](#-utilisation)
4. [Configuration Parameters](#-configuration-parameters)

## 📁 Files

### Main Files
- **`setting_scinet.py`** - Configuration and settings for SciNet
- **`train_scinet.py`** - Training function for SCINet model (dataset need to be ready at this point)

### Model Architecture
- **`model/scinet.py`** - SciNet neural network implementation

### Utilities
- **`utils/build_dataset.py`** - Custom dataset construction and preprocessing
- **`utils/data_creation_pendulum.py`** - Synthetic pendulum data generation to build the dataset
- **`utils/plot_latent_activations.py`** - Model testing and evaluation functions: mainly to have a visualization of the latent neurons' activations
- **`utils/test_scinet.py`** - Model testing and evaluation functions: mainly to have the reconstruction

### Checkpoints
- **`checkpoints/`** - Saved model checkpoints and weights during training in case of failure>

## 🔬 SCINet Overview - Architecture

SCINet is a neural network designed for recovery of the physical parameter based only on observation>
- Looks like an Auto-Encoder
- Encoder encodes in the latent space with $\mu$ and $\sigma$ in order to use a KL divergence in the loss.
- Then the input of the decoder is built with the latent vector (sampled thanks to the reparametrization trick) concatened with a question vector.
- The output must answer the question based on the pysical parameters recovered by the encoder (basically based on observation).

Note that:
- There are no physical preconceptions: the Auto-Encoder only works with observation.
- The KL divergence in the latent space force the latent variable to be independant one from another: what we want for a physical parameter.

## 🔧 Utilisation

### Synthetic data creation: damped pendulum case
```bash
cd src/magnetics_diagnostic_analysis/project_scinet/
# If using python .env
python utils/build_dataset.py
# If using uv .venv
uv run utils/build_dataset.py
```

### Train the model
```bash
cd src/magnetics_diagnostic_analysis/project_scinet/
# If using python .env
python train_scinet.py
# If using uv .venv
uv run train_scinet.py
```

### Evaluation: inference for prediction
```bash
cd src/magnetics_diagnostic_analysis/project_scinet/
# If using python .env
python utils/test_scinet.py
# If using uv .venv
uv run utils/test_scinet.py
```

### Evaluation: inference for latent space visualization
```bash
cd src/magnetics_diagnostic_analysis/project_scinet/
# If using python .env
python utils/plot_latent_activations.py
# If using uv .venv
uv run utils/plot_latent_activations.py
```

## ⚙️ Configuration Parameters

All parameters are stored in the file `setting_scinet.py`.

Key parameters for SCINet:
- **FIRST_LEARNING_RATE**: to have a stable training
- **KLD_BETA**: To recover the physical parameter but have also a good reconstruction
- **LRS_PATIENCE**: Step before that the LR automatically decreases
- **GC_MAX_NORM**: Maximum of the gradient norm during weights' update
