import numpy as np
import matplotlib.pyplot as plt

from magnetics_diagnostic_analysis.project_scinet.setting_scinet import config




def data_synthetic_pendulum(kapa, b, timesteps=50, maxtime=5.0, m=1.0, A0=1.0, phi=0.0, t: np.array = None):
    if t is None:
        t = np.linspace(0, maxtime, timesteps)
    w = np.sqrt(kapa / m) * np.sqrt(1 - b**2/(4*m*kapa))
    A = A0 * np.exp(-b * t / (2*m))

    return A * np.cos(w * t + phi)



def plot_synthetic_pendulum(timeserie, timeserienext, question, answer, timesteps=50, maxtime=5.0):
    t = np.linspace(0, maxtime, timesteps)
    tnext = np.linspace(maxtime, maxtime*2, timesteps)
    plt.figure(figsize=(10, 6))
    plt.plot(t, timeserie, label='Observation', color='blue')
    plt.plot(tnext, timeserienext, label='Prediction', color='orange')
    plt.scatter(question, answer, color='red', label='Question/Answer', zorder=5)
    plt.title(f'Synthetic Pendulum')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()
    path = config.DIR_FIGURES / "one_synthetic_pendulum.png"
    plt.savefig(path)
    plt.close()






if __name__ == "__main__":
    kapa_range = config.KAPA_RANGE
    b_range = config.B_RANGE
    kapa = np.random.uniform(*kapa_range)
    b = np.random.uniform(*b_range)

    pendulum1 = data_synthetic_pendulum(kapa, b, timesteps=config.TIMESTEPS, maxtime=config.MAXTIME)
    pendulum1next = data_synthetic_pendulum(kapa, b, t=np.linspace(config.MAXTIME, config.MAXTIME*2, config.TIMESTEPS))

    question = np.random.uniform(0, config.MAXTIME*2)
    answer = data_synthetic_pendulum(kapa, b, t=np.array(question))

    plot_synthetic_pendulum(pendulum1, pendulum1next, question, answer, timesteps=config.TIMESTEPS, maxtime=config.MAXTIME)