import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, device: str = "cpu", start_idx: int = 0, end_idx: int = -1):
        self.data = torch.from_numpy(data[start_idx:end_idx]).float()       # Convert to float32 to be compatible with PyTorch model parameters

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]



def create_data_loaders(
        data, 
        batch_size:int = 10, 
        gap_time: int = 10, 
        set_separations:list[int] = [12000, 15000], 
        device: str = "cpu"
        ) -> tuple[DataLoader]:
    """
    Create train, validation and test data loaders from time series data
    
    Args:
        data: numpy array of shape [timesteps, channels, height, width]
        batch_size: batch size for data loaders
        gap_time: time gap between samples (used for calculating indices)
        device: device to load data on
    
    Returns:
        train_loader, valid_loader, test_loader: DataLoader objects
    """
    train_end = set_separations[0] // gap_time
    valid_start = set_separations[0] // gap_time
    valid_end = set_separations[1] // gap_time
    test_start = set_separations[1] // gap_time
    total_timesteps = len(data)
    
    # Adjust indices if they exceed data length
    train_end = min(train_end, total_timesteps)
    valid_start = min(valid_start, total_timesteps)
    valid_end = min(valid_end, total_timesteps)
    test_start = min(test_start, total_timesteps)
    
    # Datasets
    train_dataset = TimeSeriesDataset(data, device, 0, train_end)
    valid_dataset = TimeSeriesDataset(data, device, valid_start, valid_end)
    test_dataset = TimeSeriesDataset(data, device, test_start, total_timesteps)
    
    # Check if batch sizes are compatible
    if len(train_dataset) % batch_size != 0:
        print(f"Warning: Batch size {batch_size} does not divide train dataset size {len(train_dataset)}")
    if len(valid_dataset) % batch_size != 0:
        print(f"Warning: Batch size {batch_size} does not divide validation dataset size {len(valid_dataset)}")
    if len(test_dataset) % batch_size != 0:
        print(f"Warning: Batch size {batch_size} does not divide test dataset size {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,      # Very important
        drop_last=False     # Either that with the assertion or drop_last=True without assertion (for more freedom)
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, valid_loader, test_loader



if __name__ == "__main__":

    # Example data (replace with your actual data loading)
    # data = np.load('your_file.npy')  # shape [2000, 3, 32, 32]
    data = np.random.randn(2000, 3, 32, 32)  # Same shape as the window_matrix
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        data=data,
        batch_size=10,
        set_separations=[12000, 15000],
        gap_time=10,
        device="cpu"  # or "cuda" if available
    )
    
    # Print dataset sizes
    print(f"Train batches: {len(train_loader)}, Total samples: {len(train_loader.dataset)}")
    print(f"Validation batches: {len(valid_loader)}, Total samples: {len(valid_loader.dataset)}")
    print(f"Test batches: {len(test_loader)}, Total samples: {len(test_loader.dataset)}")
    
    # Example of iterating through data
    for batch_idx, batch_data in enumerate(train_loader):
        print(f"Batch {batch_idx}: shape {batch_data.shape}")
        if batch_idx == 2:  # Just show first 3 batches
            break