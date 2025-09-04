import torch

# For tensor of size (batch, time_steps, n_features)
def normalize_batch(batch):
    # Compute min and max for each feature across time_steps
    min_vals = batch.amin(dim=(1), keepdim=True)
    max_vals = batch.amax(dim=(1), keepdim=True)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    # Normalize to [-1, 1]
    normalized_batch = 2 * (batch - min_vals) / ranges - 1
    return normalized_batch, min_vals, max_vals

def denormalize_batch(normalized_batch, min_vals, max_vals):
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    # Reverse the normalization: [-1, 1] -> original scale
    denormalized_batch = (normalized_batch + 1) / 2 * ranges + min_vals
    return denormalized_batch


if __name__ == "__main__":

    batch = torch.randn(10, 200, 96)  # Shape (10, 200, 96)
    print("Original batch shape:", batch.shape)

    normalized_batch, min_vals, max_vals = normalize_batch(batch)
    print("\nNormalized batch shape:", normalized_batch.shape)

    denormalized_batch = denormalize_batch(normalized_batch, min_vals, max_vals)
    print("\nDenormalized batch shape:", denormalized_batch.shape)

    # Check if denormalization is close to original
    residuals = torch.sum(torch.abs(batch - denormalized_batch))
    print("\nResiduals after denormalization:", residuals.item())