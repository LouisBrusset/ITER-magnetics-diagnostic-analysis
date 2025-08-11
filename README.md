# ITER Magnetics Diagnostic Analysis

A Python package for detecting faulty signals in magnetics diagnostics using self-supervised learning techniques. This project was developed during a 6-month internship at ITER Organization (IO).

## 🔬 Overview

This package provides tools and algorithms for analyzing magnetics diagnostic data from tokamak experiments, specifically designed to identify anomalies and faulty signals using advanced machine learning techniques including:

- **MSCRED** (Multi-Scale Convolutional Recurrent Encoder-Decoder)
- **VAE** (Variational Autoencoder) 
- Traditional anomaly detection methods

## Table of Contents

1. [Overview](#-overview)
2. [Package Structure](#-package-structure)
3. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
4. [Usage](#-usage)
   - [Quick Start](#quick-start)
   - [Data Loading](#data-loading)
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

- **`data_loading/`** - Data downloading and preprocessing utilities
- **`data_analysis/`** - Analysis tools and Jupyter notebooks
- **`ml_tools/`** - Machine learning utilities (early stopping, etc.)
- **`project_mscred/`** - MSCRED implementation for anomaly detection
- **`project_vae/`** - VAE implementation for anomaly detection

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

   #Without dev dependencies
   pip install .
   ```

## 📈 Usage

### Quick Start

```python
import magnetics_diagnostic_analysis as mda

# Get list of shots from M9 campaign
shots = mda.shot_list("M9")

# Load data for analysis
mda.load_data(
    file_path="data/",
    suffix="_analysis", 
    train_test_rate=0.8,
    shots=shots[:10],
    groups=["magnetics"],
    permanent_state=True
)

# Initialize early stopping for training
early_stop = mda.EarlyStopping(patience=10)
```

### Data Loading

```python
from magnetics_diagnostic_analysis.data_loading import shot_list, load_data

# Get shots with specific criteria
good_shots = shot_list(campaign="M9", quality=True)

# Build dataset for specific shots
from magnetics_diagnostic_analysis.data_loading import build_level_2_data_per_shot

dataset = build_level_2_data_per_shot(
    shots=good_shots[:5],
    groups=["magnetics", "efit"],
    permanent_state=True
)
```

## 🔧 Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Type Checking

```bash
mypy src/
```

## 📊 Data Source

The data comes from the MAST (Mega Amp Spherical Tokamak) experiment, accessible through:
- **MAST Data Portal**: https://mastapp.site/
- **Campaign**: M9 (primary focus)
- **Diagnostics**: Summary, Magnetics, EFM (EFIT reconstructions)

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
- École des Mines de Paris for academic supervision and providing knowlodge

## 📚 References

- [MAST Experiment Documentation](https://mastapp.site/)
- [ITER Organization](https://www.iter.org/)
- [MSCRED](https://arxiv.org/abs/1811.08055): A Deep Neural Network for Multiscale Time-series Anomaly Detection
- [VAE](https://arxiv.org/abs/1807.10300): Variational Autoencoders for Anomaly Detection
- [MAST VAE code experimentation by Samuel Jackson](https://github.com/samueljackson92/mast-signal-validation)
