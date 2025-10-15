import numpy as np
import matplotlib.pyplot as plt


def generate_multivariate_ts_2(
    n_variables: int, n_timesteps: int, noise_level: float = 0.1, seed: bool = None
) -> np.ndarray:
    """
    Improved version with amplitude control and large-scale sinusoids.

    Args:
        n_variables (int): Number of time series
        n_timesteps (int): Length of series
        noise_level (float): Noise level (standard deviation)
        seed (int): Seed for reproducibility

    Returns:
        np.array: Matrix of shape (n_variables, n_timesteps)
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.zeros((n_variables, n_timesteps), dtype=np.float32)
    time = np.arange(n_timesteps, dtype=np.float32)

    # Unique parameters for each variable
    base_amps = np.random.uniform(5, 20, n_variables)  # Amplitude de base
    slow_freqs = np.random.uniform(0.001, 0.01, n_variables)  # Fréquences lentes
    fast_freqs = np.random.uniform(0.05, 0.2, n_variables)  # Fréquences rapides
    phases = np.random.uniform(0, 2 * np.pi, n_variables)  # Déphasages

    for i in range(n_variables):
        # Primary component (slow sinusoid)
        slow_wave = base_amps[i] * np.sin(2 * np.pi * slow_freqs[i] * time + phases[i])

        # Secondary component (fast sinusoid)
        fast_wave = 0.5 * base_amps[i] * np.sin(2 * np.pi * fast_freqs[i] * time)

        # Autoregressive noise component
        # AR(1) process with a random walk component
        ar_noise = np.zeros(n_timesteps)
        ar_noise[0] = np.random.normal(0, 1)
        for t in range(1, n_timesteps):
            ar_noise[t] = 0.7 * ar_noise[t - 1] + np.random.normal(0, 0.5)

        # Combining components
        data[i] = slow_wave + fast_wave + ar_noise

    # Mesurement noise
    noise = np.random.normal(0, noise_level, (n_variables, n_timesteps))
    data = data + noise
    return (data - np.mean(data, axis=1, keepdims=True)) / np.std(
        data, axis=1, keepdims=True
    )


def generate_example():
    return generate_multivariate_ts_2(
        n_variables=32, n_timesteps=20000, noise_level=0.7, seed=42
    )


if __name__ == "__main__":

    data_train = generate_example()
    print(f"data_train.shape: {data_train.shape}")

    plt.figure(figsize=(12, 6))
    for i in range(data_train.shape[0]):
        plt.plot(data_train[i], label=f"{i+1}th serie")
    plt.title("Generated Multivariate Time Series")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.show()
