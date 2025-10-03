# Notebooks - Experiments

This directory contains Jupyter notebooks for machine learning experiments and model testing on magnetics diagnostic data.
We used notebooks to test our model because it keeps variables in cache so we can use them again, and we win time in development.

## Table of Contents

2. [Experiment Types](#-experiment-types)
1. [Available Notebooks](#-available-notebooks)
3. [Model Testing](#-model-testing)
4. [Quick Start](#-quick-start)

## 🔬 Experiment Types

- **Synthetic Data Testing**: Model validation on controlled synthetic datasets
- **Spectral Analysis**: Frequency domain feature extraction and analysis
- **Pipeline Testing**: End-to-end workflow validation
- **Model Comparison**: Performance evaluation across different architectures


## 📊 Available Notebooks

### `efit++_analysis.ipynb`
Analysis and experiments with EFIT++ equilibrium reconstruction data:
- EFIT++ data downloading from the EFM group in the FAIR-MAST dataset
- Comparison in time with magnetics diagnostics
- Analysis of seasonality and seasonal breaks
> We want to see if we can use the `fwtmp` variable to label the magnetic dataset

### `mscred_test_synthetic_data.ipynb`
MSCRED model testing on synthetic datasets:
- Synthetic data generation for testing
- MSCRED model architecture
- Anomaly detection performance evaluation
> Implementation of MSCRED architechture to spot anomalies in the timeserie

### `scinet_on_abnormal_data.ipynb`
SCINet model experiments on abnormal data patterns:
- Synthetic anomaly creation
- Abnormal data pattern analysis
- SCINet model application
- Performance evaluation on edge cases
> See if SciNet works with abnormal data

### `scinet_on_synthetic_pendulum.ipynb`
SCINet model testing on synthetic pendulum data:
- Synthetic pendulum data generation
- Scinet architecture building
- SCINet model training and testing
- Time series prediction validation
> Implementation of SciNet architecture to retrieve physical parameters

### `spectral_signature.ipynb`
Spectral analysis and signature extraction:
- Frequency domain analysis
- Spectral signature computation (FFT, Welch)
> Try to determine which signal corresponds to which diagnostic sensor

### `vae_test_loop_pipline.ipynb`
VAE model testing and evaluation pipeline:
- Complete VAE testing workflow
- Model performance evaluation
- Iterative improvement pipeline
> 



## 🧪 Model Testing

- **MSCRED**: Multi-scale convolutional recurrent encoder-decoder testing
- **VAE**: Variational autoencoder anomaly detection validation
- **SCINet**: Time series prediction and analysis experiments

## 🚀 Quick Start

1. Navigate to the experiments directory
2. Start Jupyter Lab or Jupyter Notebook
3. Select the experiment notebook of interest
4. Follow the documented experimental workflow
5. Modify parameters and configurations as needed for your experiments

## 📈 Results

Experiment results and outputs are typically saved to the `results/` directory in the project root, organized by model type and experiment configuration.