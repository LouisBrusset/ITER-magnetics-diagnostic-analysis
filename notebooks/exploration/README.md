# Notebooks - Exploration

This directory contains Jupyter notebooks for exploratory data analysis of magnetics diagnostic data from tokamak experiments.

## Table of Contents

1. [Available Notebooks](#-available-notebooks)
2. [Key Features](#-key-features)
3. [Quick Start](#-quick-start)
4. [Data Sources](#-data-sources)
5. [Analysis Outputs](#-analysis-outputs)

## 📊 Available Notebooks

### `cleaned_dataframe_opening_test.ipynb`
Testing and validation of cleaned dataframe loading procedures:
- Dataset loading verification
- Data structure validation
- Quality checks for preprocessed data

### `mast_metadata_analysis.ipynb`
Comprehensive analysis of MAST experiment metadata:
- Shot quality assessment
- Campaign statistics
- Diagnostic availability analysis
- Data completeness evaluation

### `mast_shots_choice.ipynb`
Analysis and selection criteria for MAST shots:
- Shot filtering strategies
- Quality metrics evaluation
- Selection criteria development

### `scrapped_dataframe_analysis.ipynb`
Analysis of scraped and processed dataframes:
- Data structure exploration
- Variable and channel analysis
- Statistical summaries
- Data quality assessment

## 🔧 Key Features

- **Interactive Exploration**: Step-by-step data analysis workflows
- **Metadata Analysis**: Comprehensive examination of experimental parameters
- **Quality Assessment**: Data validation and filtering procedures
- **Visualization**: Statistical plots and data distribution analysis

## 🚀 Quick Start

1. Navigate to the notebooks directory
2. Start Jupyter Lab or Jupyter Notebook
3. Open any notebook to begin exploration
4. Follow the documented workflow in each notebook

## 📈 Data Sources

- MAST tokamak experimental database
- Magnetics diagnostic measurements
- Experimental metadata and shot parameters

## 📋 Analysis Outputs

- Data quality reports
- Statistical summaries
- Visualization plots
- Processed datasets ready for ML analysis

**Key objectives addressed:**
- How many shots in the M9 campaign?
- Quality labels for each shot
- Diagnostic-specific quality assessment
- EFIT++ reconstruction analysis

## 🔧 Key Features

### Data Visualization
- Variable presence analysis
- NaN statistics by shot and variable
- Data quality heatmaps
- Time series plotting

### Statistical Analysis
- Channel distribution analysis
- Data completeness assessment
- Quality metrics calculation

## 🚀 Quick Start

To run the analysis notebooks:

```bash
# Activate your environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Start Jupyter Lab
jupyter lab

# Navigate to src/magnetics_diagnostic_analysis/data_analysis/
```

## 📈 Example Analysis Workflow

```python
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
data_train = xr.open_dataset("path/to/train_data.nc")

# Analyze variable presence
var_channel_df = pd.DataFrame(good_vars_name, columns=["full_name"])
var_channel_df[["variable", "channel"]] = var_channel_df["full_name"].str.split("::", expand=True)

# Channel count analysis
channel_counts = var_channel_df.groupby("variable").size().sort_values(ascending=False)

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(data=channel_counts.head(20))
plt.title("Top 20 Variables by Channel Count")
plt.xticks(rotation=45)
plt.show()
```

## 📊 Data Sources

The analysis works with:
- **Processed NetCDF files** from the data_loading module
- **MAST metadata** from https://mastapp.site/
- **Quality assessment results** from previous processing steps

## 🎯 Analysis Outputs

### Generated Files
- `variable_presence_*.csv` - Variable availability statistics
- `nan_stats_*.csv` - Missing data analysis
- `*.png` - Visualization outputs
- `result_lists_*.json` - Quality assessment results

### Key Metrics
- Data completeness by shot
- Variable quality scores
- Channel availability statistics
- Temporal coverage analysis

## ⚙️ Configuration

Analysis parameters can be adjusted:
- **Suffix selection**: `_mscred`, `_vae`, etc.
- **Quality thresholds**: Customizable filtering criteria
- **Visualization settings**: Plot parameters and styling

## 💡 Tips

- Use `permanent_state=True` data for steady-state analysis
- Check variable presence before training models
- Review NaN statistics to understand data quality
- Use the filtered good variables list for model inputs

## Table des matières

- [Goals](#goals)
- [Files and fonctionnalités](#files-and-functionalities)
- [Installation](#installation)

## Goals

1. First, we have an unlabelled dataset. So our very first goal is to label this dataset.

## Files and fuctionalities

- `metadata_analysis.ipynb` finds the shots for the M9 campaign, finds the labels for the whole shots and finds the labels for each diagnostic thanks to outlier detection with autoencoders.

- `outlier_detection.py` gather all the functions and classes that label the diagnostics.


## Installation