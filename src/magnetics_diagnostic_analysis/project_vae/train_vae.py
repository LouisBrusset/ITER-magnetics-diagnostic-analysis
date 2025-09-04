import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.cluster import DBSCAN

from tqdm import tqdm
from pathlib import Path
import gc

from magnetics_diagnostic_analysis.project_vae.setting_vae import config
from magnetics_diagnostic_analysis.project_vae.model.vae import LSTMBetaVAE
from magnetics_diagnostic_analysis.ml_tools.metrics import vae_loss_function, vae_reconstruction_error
from magnetics_diagnostic_analysis.ml_tools.train_callbacks import EarlyStopping, LRScheduling, GradientClipping, DropOutScheduling


def pad_sequences_smartly(batch):
    """Custom collate function to pad sequences to max length in batch"""
    sequences, lengths = zip(*batch)

    # Sort in descending order (mandatory for Truncated BPTT)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    lengths_sorted, sort_idx = torch.sort(lengths_tensor, descending=True)
    sequences_sorted = [sequences[i] for i in sort_idx]
    
    # Convert sequences to tensors
    sequence_tensors = [torch.from_numpy(seq).float() for seq in sequences_sorted]
    padded_sequences = pad_sequence(
        sequence_tensors, 
        batch_first=True, 
        padding_value=0.0
    )
    return padded_sequences, lengths_sorted



def train_one_vae(model, optimizer, loader, full_loader, n_epochs_per_iter, device, verbose):
    # Setting callbacks
    early_stopper = EarlyStopping(min_delta=0.01, patience=10)
    lr_scheduler = LRScheduling(optimizer, mode='min', factor=0.66, patience=3, min_lr=1e-6, min_delta=0.001)
    gradient_clipper = GradientClipping(max_norm=2.0)

    # Training loop
    model.train()
    if verbose:
        print(f"{'Epoch':<40} {'Loss':<20} {'mse':<20} {'kld':<20}")
    for epoch in range(n_epochs_per_iter):
        total_loss = 0
        for batch_data, batch_lengths in tqdm(loader, desc=f"Training intermediate VAE", leave=False):
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)

            optimizer.zero_grad()
            recon_batch, z_mean, z_logvar = model(batch_data, batch_lengths)
            loss, mse, kld = vae_loss_function(recon_batch, batch_data, z_mean, z_logvar, batch_lengths, beta=1.1)
            loss.backward()
            gradient_clipper.on_backward_end(model)
            optimizer.step()
            total_loss += loss.item()
            total_mse, total_kld = mse.item(), kld.item()

        epo, total_loss, total_mse, total_kld = f"Iteration {epoch + 1}/{n_epochs_per_iter}", total_loss/len(loader), total_mse/len(loader), total_kld/len(loader)
        if verbose:
            print(f"{epo:<40} {total_loss:<20} {total_mse:<20} {total_kld:<20}")
        if early_stopper.check_stop(total_loss, model):
            print(f"Early stopping at epoch {epoch + 1} with loss {total_loss:.4f}")
            print(f"Restoring best weights for model.")
            early_stopper.restore_best_weights(model)
            break
        lr_scheduler.step(total_loss)
        torch.cuda.empty_cache()
        gc.collect()

    # VAE Evaluation
    model.eval()
    reconstruction_errors = []           # or len(current_subset)
    with torch.no_grad():
        for batch_data, batch_lengths in tqdm(full_loader, desc=f"Evaluating intermediate VAE", leave=False):
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)
            
            recon_batch, _, _ = model(batch_data, batch_lengths)
            mse = vae_reconstruction_error(recon_batch, batch_data, batch_lengths)
            reconstruction_errors.append(mse.cpu())

    reconstruction_errors = torch.cat(reconstruction_errors).numpy()
    return reconstruction_errors


def find_threshold_kde(scores, alpha=0.05):
    kde = KernelDensity(kernel='gaussian', bandwidth='scott')
    kde.fit(scores.reshape(-1, 1))

    # Method 1: Based on gradient of density 
    # x = np.linspace(np.min(scores), np.max(scores), 1000)
    # log_dens = kde.score_samples(x.reshape(-1, 1))
    # density = np.exp(log_dens)
    # gradient = np.gradient(density)
    # inflection_point = x[np.argmin(gradient)]
    # Method 2: Percentile of density
    density_values = np.exp(kde.score_samples(scores.reshape(-1, 1)))
    threshold_density = np.percentile(density_values, alpha * 100)

    threshold = np.percentile(scores[density_values <= threshold_density], (1-alpha)*100)
    del kde
    return threshold

def detect_outliers_kde(scores, alpha=0.05):
    threshold = find_threshold_kde(scores, alpha)
    outlier_mask = scores > threshold
    outlier_indices = np.where(outlier_mask)[0]
    
    return outlier_indices, threshold


def train_final_vae(model, optimizer, loader, full_loader, n_epochs_per_iter, device, verbose):
    # Setting callbacks
    early_stopper = EarlyStopping(min_delta=0.01, patience=10)
    lr_scheduler = LRScheduling(optimizer, mode='min', factor=0.66, patience=3, min_lr=1e-6, min_delta=0.001)
    gradient_clipper = GradientClipping(max_norm=2.0)

    # Training loop
    model.train()
    if verbose:
        print(f"{'Epoch':<40} {'Loss':<20} {'mse':<20} {'kld':<20}")
    for epoch in range(n_epochs_per_iter):
        total_loss = 0
        for batch_data, batch_lengths in tqdm(loader, desc="Training final VAE", leave=False):
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)

            optimizer.zero_grad()
            recon_batch, z_mean, z_logvar = model(batch_data, batch_lengths)
            loss, mse, kld = vae_loss_function(recon_batch, batch_data, z_mean, z_logvar, batch_lengths, beta=1.1)
            loss.backward()
            gradient_clipper.on_backward_end(model)
            optimizer.step()
            total_loss += loss.item()
            total_mse, total_kld = mse.item(), kld.item()

        epo, total_loss, total_mse, total_kld = f"Iteration {epoch + 1}/{n_epochs_per_iter}", total_loss/len(loader), total_mse/len(loader), total_kld/len(loader)
        if verbose:
            print(f"{epo:<40} {total_loss:<20} {total_mse:<20} {total_kld:<20}")
        if early_stopper.check_stop(total_loss, model):
            print(f"Early stopping at epoch {epoch + 1} with loss {total_loss:.4f}")
            print(f"Restoring best weights for model.")
            early_stopper.restore_best_weights(model)
            break
        lr_scheduler.step(total_loss)
        torch.cuda.empty_cache()
        gc.collect()

    # Latent features for all data
    model.eval()
    with torch.no_grad():
        z_mean_all = []
        for batch_data, batch_lengths in tqdm(full_loader, desc="Extracting latent features", leave=False):
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)

            z_mean, _ = model.encoder(batch_data, batch_lengths)
            z_mean_all.append(z_mean.cpu().numpy())
        
        latent_features = np.concatenate(z_mean_all, axis=0)
    print("Latent features extracted for final VAE")
    return latent_features

def find_cluster_and_classify(latent_features, dbscan_eps, dbscan_min_samples, knn_n_neighbors):
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    clusters = dbscan.fit_predict(latent_features)
    outlier_mask = clusters == -1
    binary_labels = (dbscan.labels_ != -1).astype(int)
    print("DBSCAN Clustering Completed")

    knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors, weights='distance', metric='euclidean', algorithm='auto')
    knn.fit(latent_features, binary_labels)
    print("KNN Classifier Trained")

    del dbscan
    return knn, clusters, outlier_mask


def train_iterative_vae_pipeline(
    train_dataset: Dataset,
    n_iterations: int = 5,
    n_epochs_per_iter: int = 50,
    batch_size: int = 32,
    kde_percentile_rate: float = 0.05,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    knn_n_neighbors: int = 5,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> dict:
    
    # Model parameters
    sample_data, _ = train_dataset[0]
    print("Sample data shape:", sample_data.shape)
    input_dim = sample_data.shape[-1]
    print("Input dim:", input_dim)
    hidden_dim = 16
    latent_dim = 8
    lstm_layers = 1

    # Good and bad health indices initialization & model storage
    valid_indices = list(range(len(train_dataset)))
    all_anomaly_indices = np.array([], dtype=int)
    vae_models = []

    full_loader = DataLoader(dataset=train_dataset, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             collate_fn=pad_sequences_smartly, 
                             drop_last=False,
                             pin_memory=False,
                             num_workers=0)

    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")
        print(f"Training on {len(valid_indices)} samples...")

        # Data SubSet creation
        current_subset = Subset(train_dataset, valid_indices)
        train_loader = DataLoader(dataset=current_subset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  collate_fn=pad_sequences_smartly, 
                                  drop_last=False, 
                                  pin_memory=False,
                                  num_workers=0)

        # VAE Training
        gc.collect()
        torch.cuda.empty_cache()
        vae = LSTMBetaVAE(input_dim, hidden_dim, latent_dim, lstm_layers).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        reconstruction_errors = train_one_vae(vae, optimizer, train_loader, full_loader, n_epochs_per_iter, device, verbose=True)

        # Outlier detection with KDE
        outlier_indices, threshold = detect_outliers_kde(reconstruction_errors, alpha=kde_percentile_rate)

        # Update anomaly indices
        all_anomaly_indices = np.unique(np.concatenate([all_anomaly_indices, outlier_indices]))
        valid_indices = list(np.setdiff1d(np.arange(len(train_dataset)), all_anomaly_indices))
        
        vae_models.append(vae.state_dict())
        print(f"New anomalies detected: {len(outlier_indices)}")
        print(f"Total anomalies: {len(all_anomaly_indices)}\n")

        # Delete cache and big variables
        del vae, optimizer, reconstruction_errors, current_subset, train_loader
        torch.cuda.empty_cache()
        gc.collect()



    # Last training phase
    print("Training final model...")
    final_subset = Subset(train_dataset, valid_indices)
    final_loader = DataLoader(final_subset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=pad_sequences_smartly, 
                              drop_last=False, 
                              pin_memory=False, 
                              num_workers=0)

    final_vae = LSTMBetaVAE(input_dim, hidden_dim, latent_dim, lstm_layers).to(device)
    optimizer = torch.optim.Adam(final_vae.parameters(), lr=1e-3)

    latent_features = train_final_vae(final_vae, optimizer, final_loader, full_loader, n_epochs_per_iter, device, verbose=True)

    # Final clustering on latent space on all train_dataset, with DBScan coupled with KNN
    knn, clusters, outlier_mask = find_cluster_and_classify(latent_features, dbscan_eps, dbscan_min_samples, knn_n_neighbors)

    del final_loader, final_subset, full_loader
    gc.collect()
    return {
        'final_vae': final_vae,
        'vae_models': vae_models,
        'knn': knn,
        'anomaly_indices': all_anomaly_indices,
        'latent_features': latent_features,
        'clusters': clusters,
        'outlier_mask': outlier_mask
    }





if __name__ == "__main__":
    path_train = config.DIR_PREPROCESSED_DATA / f"dataset_ip_vae_train.pt"
    path_test = config.DIR_PREPROCESSED_DATA / f"dataset_ip_vae_test.pt"
    train_set = torch.load(path_train)
    test_set = torch.load(path_test)


    n_iterations = 1
    n_epochs_per_iter = 2
    batch_size = 1
    kde_percentile_rate = 0.05
    dbscan_eps = 0.5
    dbscan_min_samples = 5
    knn_n_neighbors = 10
    device = config.DEVICE


    results = train_iterative_vae_pipeline(
        train_dataset=train_set,
        n_iterations=n_iterations,
        n_epochs_per_iter=n_epochs_per_iter,
        batch_size=batch_size,
        kde_percentile_rate=kde_percentile_rate,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        knn_n_neighbors=knn_n_neighbors,
        device=device
    )

    final_vae = results['final_vae']
    vae_models = results['vae_models']
    knn = results['knn']
    anomaly_indices = results['anomaly_indices']
    latent_features = results['latent_features']
    clusters = results['clusters']
    outlier_mask = results['outlier_mask']