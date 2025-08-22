import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

from pathlib import Path

def generate_signature_matrix(
    data: np.ndarray,
    win_size: list[int] = [10, 30, 60],
    min_time: int = 0,
    max_time: int = None,
    gap_time: int = 10,
    normalize: bool = True,
    saving: bool = False
) -> np.ndarray:
    """
    Generates multi-scale signature matrices from multivariate time series.
    
    Args:
        data: Input data in the form (n_sensors, n_timesteps)
        win_size: List of window sizes
        min_time: Starting time point
        max_time: Ending time point (if None, uses total length)
        gap_time: Interval between segments
        normalize: If True, applies min-max normalization
        save_path: Path to save matrices (if None, no save)
    
    Returns:
        4D array of signature matrices with shape [time, len(win_sizes), n_sensors, n_sensors]
    """
    if max_time is None:
        max_time = data.shape[1]

    sensor_n = data.shape[0]
    time_steps = (max_time - min_time) // gap_time
    n_win = len(win_size)
    
    # Min-Max normalization
    if normalize:
        min_val = np.min(data, axis=1, keepdims=True)
        max_val = np.max(data, axis=1, keepdims=True)
        data = (data - min_val) / (max_val - min_val + 1e-6)
    
    # Initialize 4D array [time, n, n, 3]
    result = np.zeros((time_steps, n_win, sensor_n, sensor_n))

    print("\nStarting signature matrix generation...")
    for w_idx, win in enumerate(win_size):
        print(f"Generating signature matrices with window {win}...")
        
        for t_idx, t in enumerate(range(min_time, max_time, gap_time)):
            if t < win:  # Not enough data for the window
                mat = np.zeros((sensor_n, sensor_n))
            else:
                segment = data[:, t-win:t]
                mat = np.zeros((sensor_n, sensor_n))
                
                for i in range(sensor_n):
                    for j in range(i, sensor_n):
                        mat[i,j] = np.inner(segment[i], segment[j]) / win
                        mat[j,i] = mat[i,j]  # Symmetry
                        
            result[t_idx, w_idx, :, :] = mat
        
        if saving:
            output_path = Path(__file__).absolute().parent.parent.parent.parent.parent / "data/preprocessed/mscred/windows_matrix"
            np.save(f"{output_path}/matrix_win_{win}.npy", result[:,w_idx,:,:])

    print("Signature matrix generation completed!")
    return result

def plot_signature_matrices(
    matrix_4d: np.ndarray,
    win_sizes: list[int] = [10, 30, 60], 
    gap_time: int = 10, 
    sample_times: list[int] = [30, 100, 200],
    figsize: tuple = (12, 8),
    save_name: str = "signature_matrices_plot.png"
) -> None:
    """
    Visualizes multi-scale signature matrices from 4D array [time, len(win_sizes), n, n].
    
    Args:
        matrix_4d: 4D array of signature matrices with shape [time, len(win_sizes), n, n]
        win_sizes: List of window sizes corresponding to the win channels
        gap_time: Time step between segments
        sample_times: Times to visualize (e.g., [start, middle, end] = [max(win_sizes), len(time)/2, len(time)])
        figsize: Figure dimensions

    Returns:
        plt.Figure: The created figure object
    """
    # Entries validation
    if matrix_4d.ndim != 4:
        raise ValueError("Input matrix must be 4D array [time, len(win_sizes), n, n]")
    
    n_win = len(win_sizes)
    n_samples = len(sample_times)

    print("\nCreating signature matrix plots...")
    # Create grid with correct dimensions
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_win, n_samples + 1, 
                 width_ratios=[1]*n_samples + [0.05],
                 height_ratios=[1]*n_win)

    # Find global min/max for consistent color scaling
    vmin, vmax = np.percentile(matrix_4d, [5, 95])

    # Plot signature matrices
    for i, win_size in enumerate(win_sizes):
        for j, t in enumerate(sample_times):
            ax = plt.subplot(gs[i, j])
            
            # Calculate matrix index
            matrix_idx = t // gap_time
            if matrix_idx >= matrix_4d.shape[0]:
                matrix_idx = -1  # Use last available matrix
                
            # Get the matrix for this window size and time
            mat = matrix_4d[matrix_idx, i, :, :]
            
            # Plot matrix
            im = ax.imshow(mat, cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax.set_title(f"t = {t}\nw = {win_size}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add sensor labels on last row and first column
            if i == n_win - 1:
                ax.set_xlabel(f"Sensor {j+1}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Scale {i+1}", fontsize=9)

    # Add colorbar
    cax = plt.subplot(gs[:, -1])
    plt.colorbar(im, cax=cax, label="Correlation")
    
    plt.suptitle("Multi-scale Signature Matrices", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = Path(__file__).absolute().parent.parent.parent.parent.parent / "results/figures/mscred" / save_name
    plt.savefig(save_path)

    print(f"Signature matrix plots saved to {save_path}")
    return None




def create_signature_matrix_animation(
    matrix_4d: np.ndarray,
    window_index: int,
    win_size: int,
    gap_time: int = 10,
    save_name: str = "signature_animation.gif",
    fps: int = 10,
    figsize: tuple = (8, 6),
    dpi: int = 100
) -> None:
    """
    Creates an animation of signature matrices over time for a specific window.
    
    Args:
        matrix_4d: 4D array of signature matrices with shape [time, len(win_sizes), n, n]
        window_index: Index of the window to animate
        win_size: Window size (for title)
        gap_time: Time step between segments
        output_name: Name of the output GIF file
        fps: Frames per second for animation
        figsize: Figure dimensions
        dpi: Resolution for output
    """
    # Entries validation
    if matrix_4d.ndim != 4:
        raise ValueError("Input matrix must be 4D array [time, len(win_sizes), n, n]")
    if window_index not in [i for i in range(matrix_4d.shape[1])]:
        raise ValueError("window_index must be in window range [0, 1, ..., n-1]")
    print(f"\nCreating signature matrix animation for window {window_index}...")

    # Extract matrices and setup figure
    window_matrices = matrix_4d[:, window_index, :, :]
    n_frames = window_matrices.shape[0]
    
    # Create figure with constrained layout
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    
    # Global color scale
    vmin, vmax = np.percentile(window_matrices, [5, 95])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Create initial plot and colorbar
    im = ax.imshow(window_matrices[0], cmap='coolwarm', norm=norm)
    ax.set_title(f"Signature Matrix (Window size: {win_size})\nTime: 0")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add single colorbar (outside the axes)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation")

    def update(frame):
        """Update function for animation"""
        im.set_array(window_matrices[frame])
        ax.set_title(f"Signature Matrix (Window size: {win_size})\nTime: {frame * gap_time}")
        return [im]

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=min(n_frames, 200),  # Limit to 100 frames max for performance
        interval=1000/fps,  # ms between frames
        blit=True
    )
    
    # Register the animation with imageio
    output_path = Path(__file__).absolute().parent.parent.parent.parent.parent / "results/figures/mscred" / save_name
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Animation saved to {output_path}")
    return None



if __name__ == "__main__":
    data_train_with_anomalies = np.random.rand(32, 20000)

    win_size = [10, 30, 60]
    min_time = 0
    max_time = None
    gap_time = 10
    normalize = True

    signature_matrices = generate_signature_matrix(
            data=data_train_with_anomalies,
            win_size=win_size,
            min_time=min_time,
            max_time=max_time,
            gap_time=gap_time,
            normalize=normalize,
            saving=True
        )

    sample_times = [1000, 5000, 10000, 15000, 20000]

    plot_signature_matrices(
        matrix_4d=signature_matrices,
        win_sizes=win_size,
        gap_time=gap_time,
        sample_times=sample_times,
        figsize=(12, 8),
        save_name="signature_matrices_plot.png"
    )

    for win_idx in range(len(win_size)):
        create_signature_matrix_animation(
            matrix_4d=signature_matrices,
            window_index=win_idx,
            win_size=win_size[win_idx],
            gap_time=10,
            save_name=f"signature_animation_window{win_size[win_idx]}.gif",
            fps=5,
            figsize=(8, 6),
            dpi=100
        )

    print("\n\nAll tasks completed.")