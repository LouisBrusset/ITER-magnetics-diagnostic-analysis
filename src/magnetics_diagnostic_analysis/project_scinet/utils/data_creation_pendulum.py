import numpy as np
import matplotlib.pyplot as plt

def data_synthetic_pendulum(kapa, b, timesteps=50, maxtime=5.0, m=1.0, A0=1.0, phi=0.0, t: np.array = None):
    if t is None:
        t = np.linspace(0, maxtime, timesteps)
    w = np.sqrt(kapa / m) * np.sqrt(1 - b**2/(4*m*kapa))
    A = A0 * np.exp(-b * t / (2*m))

    return A * np.cos(w * t + phi)

def plot_synthetic_pendulum(timeserie, timesteps=50, maxtime=5.0):
    t = np.linspace(0, maxtime, timesteps)
    y = timeserie
    plt.figure(figsize=(10, 6))
    plt.plot(t, y)
    plt.title(f'Synthetic Pendulum')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()








if __name__ == "__main__":
    kapa_range = (5.0, 6.0)
    b_range = (0.2, 0.5)
    kapa = np.random.uniform(*kapa_range)
    b = np.random.uniform(*b_range)
    pendulum1 = data_synthetic_pendulum(kapa, b)
    plot_synthetic_pendulum(pendulum1)