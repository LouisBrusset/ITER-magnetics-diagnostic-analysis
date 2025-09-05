import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr

from pathlib import Path

from magnetics_diagnostic_analysis.project_vae.setting_vae import config

def find_seq_length(data: xr.Dataset) -> np.ndarray:
    # Find the length of each sequence in the dataset
    seq_indices = data['shot_index'].values
    return np.bincount(seq_indices)


class MultivariateTimeSerieDataset(Dataset):
    def __init__(self, data: xr.Dataset, n_chan_to_keep: int = 4, n_subsample: int = 10, max_length: int = 3000):
        # Group data by shot_index
        self.shot_indices = data['shot_index'].values
        self.unique_shots = np.unique(self.shot_indices)
        
        # Precompute sequences for each shot index
        self.sequences = {}
        for shot in self.unique_shots:
            mask = self.shot_indices == shot
            shot_data = []
            for var in data.data_vars:
                if var == 'shot_index':
                    continue

                if data[var].ndim == 1:
                    var_data = data[var].values[mask][:, np.newaxis]
                    if len(var_data) > max_length:
                        var_data = var_data[:max_length]

                else:
                    var_data = data[var].values[mask]
                    if var_data.shape[1] > n_chan_to_keep:
                        var_data = var_data[:, :n_chan_to_keep]
                    if len(var_data) > max_length:
                        var_data = var_data[:max_length]
                        
                var_data = var_data[::n_subsample]
                shot_data.append(var_data)
            self.sequences[shot] = np.concatenate(shot_data, axis=1)      # axis=1 => along features dimension
        
        self.lengths = {shot: len(self.sequences[shot]) for shot in self.unique_shots}

    def __len__(self):
        return len(self.unique_shots)
    
    def __getitem__(self, idx):
        shot = self.unique_shots[idx]
        return self.sequences[shot], self.lengths[shot]
    

class OneVariableTimeSerieDataset(Dataset):
    def __init__(self, data: xr.Dataset, var_name: str = "ip", chan_to_keep: None | int = 1, n_subsample: int = 12, max_length: int = 3000, normalize: bool = True):
        # Group data by shot_index
        self.shot_indices = data['shot_index'].values
        self.unique_shots = np.unique(self.shot_indices)
        
        # Precompute sequences for each shot index
        self.sequences = {}
        for shot in self.unique_shots:
            mask = self.shot_indices == shot
            data_var = data[var_name].values[mask]
            if data_var.ndim > 1:
                data_var = data_var[:, chan_to_keep]
            if len(data_var) > max_length:
                data_var = data_var[:max_length] 
            data_var = data_var[::n_subsample]
            self.sequences[shot] = data_var[:, np.newaxis]  # Add feature dimension

        self.lengths = {shot: len(self.sequences[shot]) for shot in self.unique_shots}

    def __len__(self):
        return len(self.unique_shots)
    
    def __getitem__(self, idx):
        shot = self.unique_shots[idx]
        return self.sequences[shot], self.lengths[shot]



def create_datasets(
    data: xr.Dataset,
    set_separation: int = 12000,
    total_length: int = 3500,
    rd_seed: int = 42,
    multivariate: bool = False,
    save: bool = True,
) -> tuple[Dataset]:
    """
    Create train, validation and test data loaders from time series data
    
    Args:
        data: xarray Dataset with shot_index variable
        batch_size: batch size for data loaders
        set_separation: boundarie between train and test sets
        device: device to load data on
    
    Returns:
        train_loader, valid_loader, test_loader: DataLoader objects
    """
    if save:
        path_train = config.DIR_PREPROCESSED_DATA / f"dataset_vae_train.pt"
        path_test = config.DIR_PREPROCESSED_DATA / f"dataset_vae_test.pt"
        if path_train.exists() or path_test.exists():
            print("Dataset files already exist, you must delete them first if you want to recreate them")
            return None, None

    data = data.isel(time=slice(0, total_length))

    shot_indices = data['shot_index'].values
    seq_lengths = np.bincount(shot_indices)
    unique_shots = np.unique(shot_indices)
    available_shots = unique_shots.copy()

    print("unique shots:", unique_shots.shape)

    rng = np.random.default_rng(rd_seed)

    test_shot_indices = []
    cumulative_time = 0

    while (cumulative_time < total_length - set_separation):
        shot_idx = rng.choice(available_shots, size=1, replace=False)[0]
        available_shots = available_shots[available_shots != shot_idx]

        shot_length = seq_lengths[shot_idx]
        test_shot_indices.append(shot_idx)
        cumulative_time += shot_length

    train_shot_indices = list(available_shots)
    print("Train shots:", len(train_shot_indices), "Test shots:", len(test_shot_indices))
    assert len(train_shot_indices) + len(test_shot_indices) == len(unique_shots), "Some shots are missing in the split"

    train_mask = np.isin(shot_indices, train_shot_indices)
    test_mask = np.isin(shot_indices, test_shot_indices)
    print("Train samples:", train_mask.sum(), "Test samples:", test_mask.sum())

    # Create datasets for each split
    if multivariate:
        train_dataset = MultivariateTimeSerieDataset(data.isel(time=train_mask), n_chan_to_keep=config.N_CHAN_TO_KEEP, n_subsample=config.N_SUBSAMPLE, max_length=config.MAX_LENGTH)
        test_dataset = MultivariateTimeSerieDataset(data.isel(time=test_mask), n_chan_to_keep=config.N_CHAN_TO_KEEP, n_subsample=config.N_SUBSAMPLE, max_length=config.MAX_LENGTH)
    else:
        train_dataset = OneVariableTimeSerieDataset(data.isel(time=train_mask), var_name="ip", chan_to_keep=None, n_subsample=config.N_SUBSAMPLE, max_length=config.MAX_LENGTH)
        test_dataset = OneVariableTimeSerieDataset(data.isel(time=test_mask), var_name="ip", chan_to_keep=None, n_subsample=config.N_SUBSAMPLE, max_length=config.MAX_LENGTH)


    if save:
        if not path_train.exists():
            torch.save(train_dataset, path_train)
            print(f"Saved dataset to {path_train}")
        if not path_test.exists():
            torch.save(test_dataset, path_test)
            print(f"Saved dataset to {path_test}")
    
    return train_dataset, test_dataset





if __name__ == "__main__":
    path = Path().absolute().parent.parent / "data/preprocessed/mscred/data_magnetics_mscred_cleaned.nc"
    data_all = xr.open_dataset(path)


    print("--- Test on sequence lengths: ---")
    lengths = find_seq_length(data_all)
    print(data_all['shot_index'].values)
    print(lengths)
    print("Sequence lengths:", lengths.shape)
    print("None null values:", lengths[lengths > 0].shape)
    print("Index where null values:", np.where(lengths == 0)[0])
    print(lengths[550: 560])
    print(lengths[2870: 2880])


    print("\n--- Test dataset creation: ---")
    train_set, test_set = create_datasets(data_all, set_separation=config.SET_SEPARATION, total_length=config.DATA_NUMBER, rd_seed=config.SEED, multivariate=config.MULTIVARIATE, save=True)
    print("Training set size:", len(train_set))
    print("Testing set size:", len(test_set))
    print(train_set)