# ITER Magnetics Diagnostic Analysis

A Python package for detecting faulty signals in magnetics diagnostics using self-supervised learning techniques. This project was developed during a 6-month internship at ITER Organization (IO).

## 🔬 Overview

This package provides tools and algorithms for analyzing magnetics diagnostic data from tokamak experiments, specifically designed to identify anomalies and faulty signals using advanced machine learning techniques including:

- **MSCRED** (Multi-Scale Convolutional Recurrent Encoder-Decoder)
- **Iterative VAE** (Variational Autoencoder) 
- **SCINet** (Science Network -> discovery of physical concepts)
- Traditional anomaly detection methods

## Table of Contents

1. [Overview](#-overview)
2. [Package Structure](#-package-structure)
3. [File Tree Explanation](#-file-tree-explanation)
4. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
5. [Development](#-development)
   - [Running Tests](#running-tests)
   - [Code Formatting](#code-formatting)
   - [Type Checking](#type-checking)
6. [Data Source](#-data-source)
7. [Contributing](#-contributing)
8. [License](#-license)
9. [Author](#-author)
10. [Acknowledgments](#-acknowledgments)
11. [References](#-references)

## 📦 Package Structure

The package is organized into specialized modules for different aspects of magnetics diagnostic analysis:

- **`src/magnetics_diagnostic_analysis/`** - Main package source code
  - **`data_downloading/`** - MAST database integration and data acquisition utilities
  - **`ml_tools/`** - Universal machine learning utilities (metrics, training callbacks, projections, device management)
  - **`project_mscred/`** - Multi-Scale Convolutional Recurrent Encoder-Decoder for spatio-temporal anomaly detection
  - **`project_vae/`** - Iterative β-Variational Autoencoder for outlier detection and latent space analysis
  - **`project_scinet/`** - Science Network for time series prediction and physical parameter extraction

- **`notebooks/`** - Jupyter notebooks for data exploration and experimentation
  - **`exploration/`** - Data quality assessment, shot selection, and metadata analysis
  - **`experiments/`** - Model testing, validation, and performance evaluation
  - **`result_files/`** - Generated analysis outputs and visualization results

- **`data/`** - Dataset storage and management
- **`results/`** - Model parameters, figures, and analysis outputs  
- **`tests/`** - Unit tests and integration tests for all modules
- **`docs/`** - Documentation and research papers
- **`scripts/`** - Training scripts and SLURM job configurations

## 🚀 Getting Started

### Prerequisites

- Python 3.9-3.11 (Python 3.11 recommended)
- Virtual environment manager (venv, conda, uv etc.)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LouisBrusset/ITER-magnetics-diagnostic-analysis.git
   cd ITER-magnetics-diagnostic-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using uv
   python -m pip install --user pipx
   python -m pipx ensurepath
   pipx install uv
   pipx --version
   uv --version
   uv venv --python 3.11
    
   # Using venv
   python -m venv .venv
   
   # On Windows
   source .venv\Scripts\activate
   
   # On Unix/MacOS
   source .venv/bin/activate
   ```

3. **Install the package**
   ```bash
   # Dev mod
   pip install -e .

   # Without dev dependencies
   pip install .

   # Using uv
   uv pip install -e .
   uv pip install .
   ```

## 🌳 File Tree Explanation

```
├── src/magnetics_diagnostic_analysis/           # Main package source code
│   ├── data_downloading/                        # MAST database integration
│   │   └── ...                                  # Data acquisition, steady-state filtering, data washing utilities
│   ├── ml_tools/                                # Universal ML utilities
│   │   └── ...                                  # Metrics, callbacks, projections, device management, preprocessing
│   ├── project_mscred/                          # MSCRED anomaly detection
│   │   ├── model/                               # Neural network architectures
│   │   ├── utils/                               # Data processing, evaluation, matrix generation utilities
│   │   └── checkpoints/                         # Model checkpoints and saved states
│   ├── project_vae/                             # VAE outlier detection
│   │   ├── model/                               # LSTM-based VAE architectures
│   │   └── utils/                               # Dataset building, training visualization utilities
│   └── project_scinet/                          # SCINet time series prediction
│       ├── model/                               # SCINet neural network implementation
│       ├── utils/                               # Testing, dataset building, latent space analysis
│       └── checkpoints/                         # Training checkpoints and model states
├── notebooks/                                   # Jupyter notebooks for analysis
│   ├── exploration/                             # Data quality assessment and EDA
│   ├── experiments/                             # Model testing and validation
│   └── result_files/                            # Generated analysis outputs
│       ├── all_shots_magnetics/                 # Shot-level analysis results
│       ├── efit_analysis/                       # EFIT++ reconstruction analysis
│       ├── nan_stats_magnetics/                 # Missing data statistics
│       ├── non_increasing/                      # Temporal data quality analysis
│       └── spectral_signatures/                 # Frequency domain analysis results
├── data/                                        # Dataset storage and management
│   ├── raw/                                     # Original unprocessed data
│   ├── preprocessed/                            # Cleaned and filtered data
│   │   ├── mscred/                              # MSCRED-specific preprocessing outputs
│   │   └── vae/                                 # VAE-specific dataset preparations
│   ├── processed/                               # Final model-ready datasets
│   │   └── vae/                                 # VAE training results and outputs
│   └── synthetic/                               # Artificially generated datasets
│       └── scinet/                              # Synthetic pendulum data for SCINet validation
├── results/                                     # Model outputs and analysis results
│   ├── model_params/                            # Trained model weights and parameters
│   │   ├── mscred/                              # MSCRED trained models
│   │   ├── scinet/                              # SCINet trained models
│   │   └── vae/                                 # VAE trained models
│   └── figures/                                 # Generated plots and visualizations
│       ├── mast_data/                           # MAST dataset analysis plots
│       ├── mscred/                              # MSCRED training and evaluation plots
│       ├── scinet/                              # SCINet prediction and latent space plots
│       └── vae/                                 # VAE training history and latent space visualizations
│           ├── train_histories/                 # Training progress plots
│           ├── train_densities/                 # KDE threshold analysis plots
│           └── final_vae/                       # Final model evaluation plots
├── tests/                                       # Unit tests and integration tests
│   ├── test_mscred/                             # MSCRED architecture and functionality tests
│   ├── test_vae/                                # VAE model and training tests
│   └── test_scinet/                             # SCINet implementation tests
├── docs/                                        # Documentation and research references
├── scripts/                                     # Training scripts and job configurations
│
│
│
├── .venv/                                       # Virtual environment (when created locally)
├── .git/                                        # Git version control directory
├── .pytest_cache/                               # Pytest cache directory
│
├── .gitignore                                   # Git ignore rules for excluded files
├── LICENSE                                      # Project license agreement
├── pyproject.toml                               # Python project configuration and dependencies
├── README.md                                    # Project documentation (this file)
└── uv.lock                                      # UV package manager lock file
```

### Directory Purpose Summary

**Core Source Code (`src/`)**: Contains all the main implementation modules with clear separation between universal tools (`ml_tools`), data acquisition (`data_downloading`), and specialized ML approaches (`project_*`).

**Interactive Development (`notebooks/`)**: Organized into exploration (data understanding) and experiments (model validation), with systematic result storage in `result_files/`.

**Data Management (`data/`)**: Complete data lifecycle from raw MAST downloads through preprocessing pipelines to final model-ready datasets, including synthetic data for validation.

**Output Storage (`results/`)**: Centralized storage for all model artifacts (trained weights) and generated visualizations, organized by model type and analysis stage.

**Quality Assurance (`tests/`)**: Comprehensive test coverage for all major components, ensuring reliability and maintainability of the codebase.



## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/

# Run specific test modules
pytest tests/test_mscred/
pytest tests/test_vae/
pytest tests/test_scinet/
```

### Code Formatting

```bash
# Format code (if using black)
black src/

# Check code style (if using flake8)
flake8 src/
```


### Type Checking

```bash
mypy src/
```


## 📊 Data Source

The data comes from the MAST (Mega Amp Spherical Tokamak) experiment, accessible through:
- **MAST Data Portal**: https://mastapp.site/
- **Diagnostics**: Summary, Pulse_schedule, Magnetics, EFM (EFIT reconstructions)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👨‍💻 Author

**Louis Brusset**
- Email: louis.brusset@etu.minesparis.psl.eu
- Institution: École Nationale Supérieure des Mines de Paris
- Organization: ITER Organization (IO)

## 🙏 Acknowledgments

- ITER Organization for providing the internship opportunity
- MAST team for data access and support
- École des Mines de Paris for academic supervision and providing knowledge

## 📚 References

- [MAST Experiment Documentation](https://mastapp.site/)
- [ITER Organization](https://www.iter.org/)
- [MSCRED](https://arxiv.org/abs/1811.08055): A Deep Neural Network for Multiscale Time-series Anomaly Detection
- [VAE](https://arxiv.org/abs/1807.10300): Variational Autoencoders for Anomaly Detection
- [MAST VAE code experimentation by Samuel Jackson](https://github.com/samueljackson92/mast-signal-validation)
