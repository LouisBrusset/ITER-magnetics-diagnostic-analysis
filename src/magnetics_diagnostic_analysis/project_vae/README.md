# Project VAE Module

Variational Autoencoder (VAE) implementation for anomaly detection in magnetics diagnostic data.

## Table of Contents

1. [Files](#-files)
2. [VAE Overview](#-vae-overview)
3. [Mathematical Foundation](#-mathematical-foundation)
4. [Core Components](#-core-components)
5. [Usage Example](#-usage-example)
6. [Model Architecture](#-model-architecture)
7. [Training and Evaluation](#-training-and-evaluation)
8. [Advanced Configurations](#-advanced-configurations)
9. [Best Practices](#-best-practices)

## 📁 Files

### Main Files
- **`setting_vae.py`** - Configuration and settings for VAE
- **`train_vae.py`** - Training pipeline for VAE model

### Model Architecture
- **`model/`** - VAE neural network implementation and components

### Utilities
- **`utils/data_scraping.py`** - Data preparation and scraping utilities for VAE
- **`utils/dataset_building.py`** - Dataset construction and preprocessing
- **`utils/evaluation_vae.py`** - Model evaluation and testing functions
- **`utils/plot_training.py`** - Training visualization utilities

### Checkpoints
- **`checkpoints/`** - Saved model checkpoints and weights

## 🔬 VAE Overview

Variational Autoencoders (VAE) are probabilistic generative models that learn a compressed representation of normal data patterns. For anomaly detection in magnetics diagnostics:

- **Encoder**: Maps input data to latent probability distributions
- **Decoder**: Reconstructs data from sampled latent representations
- **Probabilistic Framework**: Provides uncertainty estimates for predictions
- **Anomaly Detection**: Based on reconstruction probability and latent space positioning

## 🧮 Mathematical Foundation

### VAE Loss Function
```
L_VAE = L_reconstruction + β × L_KL
```

Where:
- **L_reconstruction**: How well the model reconstructs input data
- **L_KL**: Kullback-Leibler divergence regularization term

## 🔧 Core Components

### Model Training
```python
from magnetics_diagnostic_analysis.project_vae import train_vae

# Train VAE model
train_vae.main()
```

### Configuration
```python
from magnetics_diagnostic_analysis.project_vae import setting_vae

# Load VAE settings
config = setting_vae.load_config()
```

### Data Processing
```python
from magnetics_diagnostic_analysis.project_vae.utils import data_scraping, dataset_building

# Prepare data for VAE training
data = data_scraping.load_data()
dataset = dataset_building.build_dataset(data)
```

### Evaluation and Visualization
```python
from magnetics_diagnostic_analysis.project_vae.utils import evaluation_vae, plot_training

# Evaluate trained model
results = evaluation_vae.evaluate_model(model, test_data)

# Visualize training progress
plot_training.plot_losses(training_history)
```
- **β**: Weighting parameter (β-VAE for disentanglement)

### Anomaly Score
```
Anomaly_Score = -log p(x|z) + α × ||z - μ_normal||²
```

Where:
- **p(x|z)**: Reconstruction likelihood
- **z**: Latent representation
- **μ_normal**: Mean of normal samples in latent space

## 🔧 Key Components

### Encoder Network
```python
class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
            
        self.encoder = nn.Sequential(*layers)
        
        # Latent space parameters
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
```

### Decoder Network
```python
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims=[64, 128]):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
```

## 🚀 Usage Example

### Basic VAE Training
```python
import torch
import torch.nn as nn
from magnetics_diagnostic_analysis.ml_tools import EarlyStopping

# Load and preprocess data
train_data = load_magnetics_data(permanent_state=True)
val_data = load_validation_data()

# Initialize VAE
vae = VariationalAutoencoder(
    input_dim=train_data.shape[-1],
    latent_dim=20,
    beta=1.0
)

# Training setup
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
early_stop = EarlyStopping(patience=10, min_delta=1e-4)

# Training loop
for epoch in range(epochs):
    train_loss = train_vae(vae, train_data, optimizer)
    val_loss = validate_vae(vae, val_data)
    
    if early_stop.check_stop(val_loss, vae):
        break

# Anomaly detection
anomaly_scores = compute_anomaly_scores(vae, test_data)
```

## 📊 Data Preprocessing for VAE

### Specific Requirements
```python
def preprocess_for_vae(magnetics_data):
    """
    Preprocess magnetics data for VAE training.
    
    Steps:
    1. Permanent state filtering
    2. Normalization (important for VAE)
    3. Sliding window creation
    4. Missing value handling
    """
    
    # 1. Filter permanent state periods
    filtered_data = apply_permanent_state_filter(magnetics_data)
    
    # 2. Standard normalization (critical for VAE)
    normalized_data = (filtered_data - filtered_data.mean()) / filtered_data.std()
    
    # 3. Create sliding windows
    windows = create_sliding_windows(normalized_data, window_size=100, stride=50)
    
    # 4. Handle missing values
    clean_windows = handle_missing_values(windows, strategy='interpolation')
    
    return clean_windows
```

## 🎯 Anomaly Detection Pipeline

### 1. Training Phase
```python
def train_vae_anomaly_detector(normal_data):
    """Train VAE on normal magnetics data only."""
    
    # Preprocess normal data
    processed_data = preprocess_for_vae(normal_data)
    
    # Train VAE
    vae = train_vae(processed_data)
    
    # Compute normal data statistics
    normal_scores = compute_reconstruction_errors(vae, processed_data)
    threshold = np.percentile(normal_scores, 95)  # 95th percentile
    
    return vae, threshold
```

### 2. Detection Phase
```python
def detect_anomalies_vae(vae, test_data, threshold):
    """Detect anomalies in test data."""
    
    # Preprocess test data
    processed_test = preprocess_for_vae(test_data)
    
    # Compute anomaly scores
    reconstruction_errors = compute_reconstruction_errors(vae, processed_test)
    kl_divergences = compute_kl_divergences(vae, processed_test)
    
    # Combined anomaly score
    anomaly_scores = reconstruction_errors + 0.1 * kl_divergences
    
    # Classify anomalies
    anomalies = anomaly_scores > threshold
    
    return anomalies, anomaly_scores
```

## 🔍 Interpretability Features

### Latent Space Analysis
```python
def analyze_latent_space(vae, data, labels):
    """Analyze the learned latent representations."""
    
    # Encode data to latent space
    with torch.no_grad():
        mu, logvar = vae.encode(data)
        z = vae.reparameterize(mu, logvar)
    
    # Visualize latent space (2D projection)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z.cpu().numpy())
    
    # Plot normal vs anomalous in latent space
    plt.scatter(z_2d[labels==0, 0], z_2d[labels==0, 1], label='Normal', alpha=0.6)
    plt.scatter(z_2d[labels==1, 0], z_2d[labels==1, 1], label='Anomaly', alpha=0.6)
    plt.legend()
    plt.title('Latent Space Representation')
```

### Feature Attribution
```python
def compute_feature_importance(vae, anomalous_sample):
    """Identify which magnetics channels contribute most to anomaly."""
    
    # Compute gradients of reconstruction error w.r.t. input
    anomalous_sample.requires_grad_(True)
    
    reconstruction = vae(anomalous_sample)
    reconstruction_error = F.mse_loss(reconstruction, anomalous_sample)
    
    gradients = torch.autograd.grad(reconstruction_error, anomalous_sample)[0]
    importance_scores = gradients.abs().mean(dim=0)
    
    return importance_scores
```

## 🔧 Advanced Configurations

### β-VAE for Disentanglement
```python
# Higher β encourages disentangled representations
beta_vae = VariationalAutoencoder(
    input_dim=n_channels,
    latent_dim=32,
    beta=4.0  # Higher beta for more disentanglement
)
```

### Conditional VAE
```python
# Condition on shot metadata (campaign, discharge conditions)
cvae = ConditionalVAE(
    input_dim=n_channels,
    condition_dim=metadata_dim,
    latent_dim=20
)
```

## 📈 Integration with Magnetics Analysis

### Multi-Shot Analysis
```python
def analyze_shot_anomalies(vae, shots_data):
    """Analyze anomalies across multiple shots."""
    
    shot_anomaly_rates = {}
    
    for shot_id, shot_data in shots_data.items():
        anomalies, scores = detect_anomalies_vae(vae, shot_data, threshold)
        anomaly_rate = anomalies.sum() / len(anomalies)
        shot_anomaly_rates[shot_id] = anomaly_rate
    
    return shot_anomaly_rates
```

### Channel-Specific Analysis
```python
def identify_problematic_channels(vae, anomalous_data):
    """Identify which magnetics channels are most problematic."""
    
    channel_importance = {}
    
    for i, channel in enumerate(channel_names):
        importance = compute_feature_importance(vae, anomalous_data)
        channel_importance[channel] = importance[i].item()
    
    return sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
```

## 💡 Best Practices

- **Normalization**: Critical for VAE convergence
- **Latent Dimension**: Balance between expressivity and regularization
- **β Parameter**: Tune for your specific anomaly detection needs  
- **Training Data**: Use only high-quality, normal operation data
- **Threshold Selection**: Use validation set with known anomalies if available
- **Regular Retraining**: Update model as new normal data becomes available
