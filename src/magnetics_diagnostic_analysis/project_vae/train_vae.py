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
from magnetics_diagnostic_analysis.ml_tools.preprocessing import normalize_batch, denormalize_batch
from magnetics_diagnostic_analysis.project_vae.utils.dataset_building import MultivariateTimeSerieDataset, OneVariableTimeSerieDataset
from magnetics_diagnostic_analysis.project_vae.utils.plot_training import plot_history, plot_density_and_threshold, plot_projected_latent_space, plot_random_reconstructions


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
    history = []
    model.train()
    if verbose:
        print(f"{'Epoch':<30} {'Loss':<25} {'mse':<25} {'kld':<25}")
    for epoch in range(n_epochs_per_iter):
        total_loss, total_mse, total_kld = 0, 0, 0
        for batch_data, batch_lengths in tqdm(loader, desc=f"Training intermediate VAE", leave=False):
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)
            normalized_batch, _, _ = normalize_batch(batch_data)

            optimizer.zero_grad()
            recon_batch, z_mean, z_logvar = model(normalized_batch, batch_lengths)
            loss, mse, kld = vae_loss_function(recon_batch, normalized_batch, z_mean, z_logvar, batch_lengths, beta=config.BETA)
            loss.backward()
            gradient_clipper.on_backward_end(model)
            optimizer.step()
            total_loss += loss.item()
            total_mse, total_kld = total_mse+mse.item(), total_kld+kld.item()
            history.append(loss.item())

        epo, total_loss, total_mse, total_kld = f"Iteration {epoch + 1}/{n_epochs_per_iter}", total_loss/len(loader), total_mse/len(loader), total_kld/len(loader)
        if verbose:
            print(f"{epo:<30} {total_loss:<25} {total_mse:<25} {total_kld:<25}")
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
            normalized_batch, _, _ = normalize_batch(batch_data)

            recon_batch, _, _ = model(normalized_batch, batch_lengths)
            mse = vae_reconstruction_error(recon_batch, normalized_batch, batch_lengths)
            reconstruction_errors.append(mse.cpu())

    reconstruction_errors = torch.cat(reconstruction_errors).numpy()


    return history, reconstruction_errors


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
    return threshold, density_values

def detect_outliers_kde(scores, alpha=0.05):
    threshold, density_values = find_threshold_kde(scores, alpha)
    outlier_mask = scores > threshold
    outlier_indices = np.where(outlier_mask)[0]
    
    return outlier_indices, threshold, density_values


def pad_results(x_couples: list) -> list:
    """Pad results to the max length in the list"""
    print(x_couples[0].shape)
    max_length = max([x.shape[2] for x in x_couples])
    print("Max length:", max_length)
    padded_results = []
    for x in x_couples:
        pad_size = max_length - x.shape[2]
        if pad_size > 0:
            pad_tensor = np.zeros((x.shape[0], x.shape[1], pad_size, x.shape[3]))
            padded_x = np.concatenate([x, pad_tensor], axis=2)
            print("Padded shape:", padded_x.shape)
        else:
            padded_x = x
        if padded_x.shape[2] != max_length:
            raise ValueError("Padding failed, shapes do not match")
        padded_results.append(padded_x)
    return padded_results


def train_final_vae(model, optimizer, loader, full_loader, n_epochs_per_iter, device, verbose):
    # Setting callbacks
    early_stopper = EarlyStopping(min_delta=0.01, patience=10)
    lr_scheduler = LRScheduling(optimizer, mode='min', factor=0.66, patience=3, min_lr=1e-6, min_delta=0.001)
    gradient_clipper = GradientClipping(max_norm=2.0)

    # Training loop
    history = []
    model.train()
    if verbose:
        print(f"{'Epoch':<30} {'Loss':<25} {'mse':<25} {'kld':<205}")
    for epoch in range(n_epochs_per_iter):
        total_loss, total_mse, total_kld = 0, 0, 0
        for batch_data, batch_lengths in tqdm(loader, desc="Training final VAE", leave=False):
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)
            normalized_batch, _, _ = normalize_batch(batch_data)

            optimizer.zero_grad()
            recon_batch, z_mean, z_logvar = model(normalized_batch, batch_lengths)
            loss, mse, kld = vae_loss_function(recon_batch, normalized_batch, z_mean, z_logvar, batch_lengths, beta=config.BETA)
            loss.backward()
            gradient_clipper.on_backward_end(model)
            optimizer.step()
            total_loss += loss.item()
            total_mse, total_kld = total_mse + mse.item(), total_kld + kld.item()
            history.append(loss.item())

        epo, total_loss, total_mse, total_kld = f"Iteration {epoch + 1}/{n_epochs_per_iter}", total_loss/len(loader), total_mse/len(loader), total_kld/len(loader)
        if verbose:
            print(f"{epo:<30} {total_loss:<25} {total_mse:<25} {total_kld:<25}")
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
        x_couple_all = []
        i=0
        for batch_data, batch_lengths in tqdm(full_loader, desc="Extracting latent features", leave=False):
            i+=1
            print(i)
            print(batch_data.shape)
            batch_data = batch_data.to(device)
            batch_lengths = batch_lengths.to(device)
            normalized_batch, min_batch, max_batch = normalize_batch(batch_data)

            x_recon, z_mean, _ = model(normalized_batch, batch_lengths)
            z_mean_all.append(z_mean.cpu().numpy())

            x_recon = denormalize_batch(x_recon, min_batch, max_batch)
            x_couple = np.concatenate([batch_data.unsqueeze(0).cpu().numpy(), x_recon.unsqueeze(0).cpu().numpy()], axis=0)
            x_couple_all.append(x_couple)

        latent_features = np.concatenate(z_mean_all, axis=0)
        reconstruction_couples = np.concatenate(pad_results(x_couple_all), axis=1)

    print("Latent features extracted for final VAE")
    return history, reconstruction_couples, latent_features

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
    hidden_dim = config.LSTM_HIDDEN_DIM
    latent_dim = config.LATENT_DIM
    lstm_layers = config.LSTM_NUM_LAYERS

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
        optimizer = torch.optim.Adam(vae.parameters(), lr=config.FIRST_LEARNING_RATE)

        history, reconstruction_errors = train_one_vae(vae, optimizer, train_loader, full_loader, n_epochs_per_iter, device, verbose=True)
        plot_history(history, save_path= config.DIR_FIGURES / f"train_histories/vae_training_iteration_{iteration + 1}.png")
        # Outlier detection with KDE
        outlier_indices, threshold, density_values = detect_outliers_kde(reconstruction_errors, alpha=kde_percentile_rate)
        plot_density_and_threshold(density_values, threshold, save_path= config.DIR_FIGURES / f"train_densities/kde_threshold_iteration_{iteration + 1}.png")

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
    optimizer = torch.optim.Adam(final_vae.parameters(), lr=config.FIRST_LEARNING_RATE)

    history, reconstruction_couples, latent_features = train_final_vae(final_vae, optimizer, final_loader, full_loader, n_epochs_per_iter, device, verbose=True)
    plot_history(history, save_path= config.DIR_FIGURES / f"final_vae/vae_training_final.png")
    plot_random_reconstructions(reconstruction_couples, n_samples=5, save_path= config.DIR_FIGURES / f"final_vae/random_reconstructions_final.png")

    # Final clustering on latent space on all train_dataset, with DBScan coupled with KNN
    knn, clusters, outlier_mask = find_cluster_and_classify(latent_features, dbscan_eps, dbscan_min_samples, knn_n_neighbors)
    plot_projected_latent_space(latent_features, clusters, outlier_mask, save_path= config.DIR_FIGURES / f"final_vae/latent_space_final.png")

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

def main():
    path_train = config.DIR_PREPROCESSED_DATA / f"dataset_vae_train.pt"
    #path_test = config.DIR_PREPROCESSED_DATA / f"dataset_vae_test.pt"
    train_set = torch.load(path_train)
    #test_set = torch.load(path_test)


    n_iterations = config.N_ITERATIONS
    n_epochs_per_iter = config.N_EPOCHS
    batch_size = config.BATCH_SIZE
    kde_percentile_rate = config.KDE_PERCENTILE_RATE
    dbscan_eps = config.DBSCAN_EPS
    dbscan_min_samples = config.DBSCAN_MIN_SAMPLES
    knn_n_neighbors = config.KNN_N_NEIGHBORS
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

    # Saving result to a json file
    save_path = config.DIR_PROCESSED_DATA / f"vae_iterative_results.pth"
    torch.save(results, save_path)
    print(f"Results saved to {save_path}")




if __name__ == "__main__":
    torch.backends.cudnn.enabled = False        # issue to solve in the future
    main()

