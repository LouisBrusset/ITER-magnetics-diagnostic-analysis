import numpy as np
import matplotlib.pyplot as plt

def create_anomalies(data, start_index=15000, duration_range=(50, 200), 
                    n_anomalies=5, anomaly_strength=1.5, seed=None):
    """
    Crée plusieurs anomalies dans les séries temporelles multivariées.
    
    Args:
        data: Matrice de shape (n_variables, n_timesteps)
        start_index: Index de début des anomalies
        duration_range: Tuple (min_duration, max_duration) pour la longueur des anomalies
        n_anomalies: Nombre d'anomalies à créer
        anomaly_strength: Force de l'anomalie (multiplicateur de l'écart-type)
        seed: Seed pour la reproductibilité
        
    Returns:
        np.array: Données avec anomalies
        list: Liste des informations sur les anomalies créées
    """
    if seed is not None:
        np.random.seed(seed)
    
    data_with_anomalies = data.copy()
    n_variables, n_timesteps = data.shape
    anomalies_info = []
    
    for i in range(n_anomalies):
        start_anomaly = np.random.randint(start_index, n_timesteps)
        series_num = np.random.randint(0, n_variables)
        duration = np.random.randint(duration_range[0], duration_range[1])

        end_anomaly = min(start_anomaly + duration, n_timesteps)
        actual_duration = end_anomaly - start_anomaly

        if actual_duration <= 0:
            i -= 1
            continue

        local_std = np.std(data[series_num, start_anomaly-100:start_anomaly])
        anomaly_type = np.random.choice(['spike', 'drift', 'level_shift', 'seasonal_break'])
        
        if anomaly_type == 'spike':
            spike_value = anomaly_strength * local_std * np.random.choice([-1, 1])
            data_with_anomalies[series_num, start_anomaly:end_anomaly] += spike_value

        elif anomaly_type == 'drift':
            drift_slope = anomaly_strength * local_std * 0.01 * np.random.choice([-1, 1])
            drift_values = np.arange(actual_duration) * drift_slope
            data_with_anomalies[series_num, start_anomaly:end_anomaly] += drift_values

        elif anomaly_type == 'level_shift':
            shift_value = anomaly_strength * local_std * np.random.choice([-1, 1])
            data_with_anomalies[series_num, start_anomaly:end_anomaly] += shift_value

        elif anomaly_type == 'seasonal_break':
            seasonal_amp = anomaly_strength * local_std * 0.5
            freq_change = np.random.uniform(0.5, 2.0)
            time_segment = np.arange(actual_duration)
            seasonal_anomaly = seasonal_amp * np.sin(2 * np.pi * freq_change * time_segment / 100)
            data_with_anomalies[series_num, start_anomaly:end_anomaly] += seasonal_anomaly

        anomalies_info.append({
            'series': series_num,
            'start_index': start_anomaly,
            'duration': actual_duration,
            'type': anomaly_type,
            'strength': anomaly_strength
        })
    
    return data_with_anomalies, anomalies_info

def plot_anomalies(data_normal, data_anomalous, anomalies_info, n_series_to_plot=3):
    """
    Visualise les séries avec anomalies.
    """
    n_variables, n_timesteps = data_normal.shape
    
    fig, axes = plt.subplots(n_series_to_plot, 1, figsize=(12, 3*n_series_to_plot))
    if n_series_to_plot == 1:
        axes = [axes]
    
    series_to_plot = np.random.choice(n_variables, n_series_to_plot, replace=False)
    
    for i, series_num in enumerate(series_to_plot):
        axes[i].plot(data_normal[series_num], 'b-', alpha=0.7, label='Normal')
        axes[i].plot(data_anomalous[series_num], 'r-', alpha=0.9, label='With Anomalies')
        
        for anomaly in anomalies_info:
            if anomaly['series'] == series_num:
                start = anomaly['start_index']
                end = start + anomaly['duration']
                axes[i].axvspan(start, end, color='yellow', alpha=0.3, label='Anomaly' if i == 0 else "")
        
        axes[i].set_title(f'Série {series_num}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def generate_example(data_train):
    return create_anomalies(
    data=data_train,
    start_index=15000,
    duration_range=(80, 150),
    n_anomalies=8,
    anomaly_strength=2.0,
    seed=43
)


if __name__ == "__main__":
    data_train = np.random.randn(32, 20000)
    data_val_anomalous, anomalies_info = generate_example(data_train)

    print("data_val_anomalous.shape:", data_val_anomalous.shape)
    print("anomalies_info.shape:", anomalies_info.shape)

    plot_anomalies(data_train, data_val_anomalous, anomalies_info, n_series_to_plot=3)
