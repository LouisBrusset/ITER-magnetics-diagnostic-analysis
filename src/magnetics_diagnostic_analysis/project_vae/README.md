# Project VAE Module

Variational Autoencoder (VAE) implementation for anomaly detection in magnetics diagnostic data. The idea is to iterate with VAE reconstruction training, then outliers elimination, and finaly clusterisation of the obtained latent space.

## Table of Contents

1. [Files](#files)
2. [VAE Overview - Architecture](#vae-overview---architecture)
3. [Utilisation](#utilisation)
4. [Configuration Parameters](#configuration-parameters)

## Files

### Main Files
- **`__init__.py`** - Prepare importation for future utilisation
- **`setting_vae.py`** - Configuration and settings for VAE
- **`train_vae.py`** - Training and evaluation pipeline for VAE model

### Model Architecture
- **`model/lstm_vae.py`** - LSTM-based $\beta$-VAE neural network implementation

### Utilities
- **`utils/dataset_building.py`** - Custom dataset construction and preprocessing
- **`utils/plot_training.py`** - Training visualization utilities

### Checkpoints
- **`checkpoints/`** - Saved model checkpoints and weights at each epoch during training

## VAE Overview - Architecture

Variational Autoencoders (VAE) are probabilistic generative models that learn a compressed representation of normal data patterns. For anomaly detection in magnetics diagnostics:

- **Encoder**: Maps input data to latent probability distributions
- **Decoder**: Reconstructs data from sampled latent representations
- **KLD Loss**: The loss is not only based on the reconstruction errors but also on the discrepencies between the latent variables distribution and a reference distribution (multivariate gaussian ~ N(0,I))

VAE are really good in anomaly detection. To improve this anomaly detection, we iterate on VAEs. Here is the workflow:

1. Train a $\beta$-VAE on trainset (trainset of the iteration)
2. Perform validation on the trainset (full trainset)
3. Select bad reconstructions as outliers thanks to a KDE model
4. Delete those outliers from the trainset and return to 1
5. After some iterations, train a final $\beta$-VAE on the full set
6. Infer to found latent variables
7. Clusterize those variable to spot groups in the dataset
8. Project the high-dimensional latent space into a 2- or 3-dimensional space thanks to UMAP.



## Utilisation

### Data downloading and cleaning

See the `src/magnetics_diagnostic/analysis/project_mscred/` project and its `README.md` file.

### Dataset building
```bash
cd src/magnetics_diagnostic_analysis/project_vae/
# If using python .env
python utils/dataset_building.py
# If using uv .venv
uv run utils/dataset_building.py
```

### Train model 
```bash
cd src/magnetics_diagnostic_analysis/project_vae/
# If using python .env
python train_vae.py
# If using uv .venv
uv run train_vae.py
```



## Configuration Parameters

All parameters are stored in the file `setting_vae.py`.

Key parameters for Iterative $\beta$-VAEs:
- **MULTIVARIATE**: Is the input timeserie used multivariate or not
    - **N_CHAN_TO_KEEP** or **VAR_NAME**: Whether MULTIVARIATE is respectivly `True` or `False`.
- **MAX_LENGHT and N_SUBSAMPLE**: Handle the lenght of the input to prevent a to big BPTT (BackPropagatino Throught Time)
- **STEADY_STATE**: Keep the full timeserie (withour selecting one part)

- **FIRST_LEARNING_RATE**: to have a stable training
- **KLD_BETA**: Trad-off between recovering meaningful parameter but having also a good reconstruction
- **LRS_PATIENCE**: Step before that the LR automatically decreases

- **KDE_PERCENTILE_RATE**: Wheter we want to exclude lots of outliers per iteration or not.
- **DBSCAN_EPS and DBSCAN_MIN_SAMPLES**: Hyperparameters for DBScan
- **KNN_N_NEIGHBORS**: Hyperparameter for KNN
