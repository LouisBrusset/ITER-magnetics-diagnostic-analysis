import numpy as np
import matplotlib.pyplot as plt

from magnetics_diagnostic_analysis.project_scinet.setting_scinet import config


def data_synthetic_pendulum(
    kapa: float,
    b: float,
    timesteps: int = 50,
    maxtime: float = 5.0,
    m: float = 1.0,
    A0: float = 1.0,
    phi: float = 0.0,
    t: np.array = None,
) -> np.array:
    """
    Generate synthetic pendulum data based on the damped harmonic oscillator model.

    Args:
        kapa (float): Spring constant.
        b (float): Damping coefficient.
        timesteps (int): Number of time steps to generate. Default is 50.
        maxtime (float): Maximum time value. Default is 5.0.
        m (float): Mass of the pendulum. Default is 1.0.
        A0 (float): Initial amplitude. Default is 1.0.
        phi (float): Phase shift. Default is 0.0.
        t (np.array, optional): Specific time points to evaluate. If None, generates linearly spaced time points.

    Returns:
        np.array: The generated pendulum time series data.
    """
    if t is None:
        t = np.linspace(0, maxtime, timesteps)
    w = np.sqrt(kapa / m) * np.sqrt(1 - b**2 / (4 * m * kapa))
    A = A0 * np.exp(-b * t / (2 * m))
    return A * np.cos(w * t + phi)


def plot_synthetic_pendulum(
    timeserie: np.array,
    timeserienext: np.array,
    question: float,
    answer: float,
    timesteps: int = 50,
    maxtime: float = 5.0,
) -> None:
    """
    Plot the synthetic pendulum data.

    Args:
        timeserie (np.array): The observed time series data.
        timeserienext (np.array): The predicted time series data.
        question (float): The question input.
        answer (float): The true answer.
        timesteps (int, optional): The number of time steps. Defaults to 50.
        maxtime (float, optional): The maximum time. Defaults to 5.0.

    Returns:
        None
    """
    t = np.linspace(0, maxtime, timesteps)
    tnext = np.linspace(maxtime, maxtime * 2, timesteps)
    plt.figure(figsize=(10, 6))
    plt.plot(t, timeserie, label="Observation", color="blue")
    plt.plot(tnext, timeserienext, label="Prediction", color="orange")
    plt.scatter(question, answer, color="red", label="Question/Answer", zorder=5)
    plt.title(f"Synthetic Pendulum")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    path = config.DIR_FIGURES / "one_synthetic_pendulum.png"
    plt.savefig(path)
    plt.close()
    return None


if __name__ == "__main__":
    kapa_range = config.KAPA_RANGE
    b_range = config.B_RANGE
    kapa = np.random.uniform(*kapa_range)
    b = np.random.uniform(*b_range)

    pendulum1 = data_synthetic_pendulum(
        kapa, b, timesteps=config.TIMESTEPS, maxtime=config.MAXTIME
    )
    pendulum1next = data_synthetic_pendulum(
        kapa, b, t=np.linspace(config.MAXTIME, config.MAXTIME * 2, config.TIMESTEPS)
    )

    question = np.random.uniform(0, config.MAXTIME * 2)
    answer = data_synthetic_pendulum(kapa, b, t=np.array(question))

    plot_synthetic_pendulum(
        pendulum1,
        pendulum1next,
        question,
        answer,
        timesteps=config.TIMESTEPS,
        maxtime=config.MAXTIME,
    )
