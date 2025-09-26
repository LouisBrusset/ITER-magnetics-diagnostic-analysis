import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import gc

from magnetics_diagnostic_analysis.project_scinet.setting_scinet import config
from magnetics_diagnostic_analysis.project_scinet.utils.data_creation_pendulum import data_synthetic_pendulum
from magnetics_diagnostic_analysis.project_scinet.utils.build_dataset import build_dataset
from magnetics_diagnostic_analysis.project_scinet.model.scinet import PendulumNet




def make_one_prediction(model: nn.Module, observation: np.array, question: float, device: torch.device = torch.device('cpu')) -> float:
    torch.cuda.empty_cache()
    model.to(device).eval()
    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
    question_tensor = torch.tensor([[question]], dtype=torch.float32).to(device)
    with torch.no_grad():
        possible_answer, _, _ = model(observation_tensor, question_tensor)
    return possible_answer.item()

def make_timeserie_prediction(model: nn.Module, observation: np.array, questions: np.array, device: torch.device = torch.device('cpu')) -> float:
    torch.cuda.empty_cache()
    model.to(device).eval()
    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
    possible_answers = []
    with torch.no_grad():
        for q in questions:
            question_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            possible_answer, _, _ = model(observation_tensor, question_tensor)
            possible_answers.append(possible_answer.item())
    return np.array(possible_answers)


def plot_one_prediction(observation, question, answer, possible_answer, maxtime, timesteps) -> None:
    fig = plt.figure(figsize=(10, 6))
    time = np.linspace(0, maxtime, timesteps)
    plt.plot(time, observation, label='Observation', color='blue')
    plt.scatter(question, possible_answer, color='red', label='Prediction', zorder=5)
    plt.scatter(question, answer, color='green', label='True Answer', zorder=5)
    plt.title('Pendulum Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    path = config.DIR_FIGURES / "scinet_pendulum_one_prediction.png"
    plt.savefig(path)
    plt.close()    
    return None

def plot_timeserie_prediction(observations, answers, possible_answers, maxtime, timesteps) -> None:
    fig = plt.figure(figsize=(10, 6))
    time = np.linspace(0, maxtime, timesteps)
    time_next = np.linspace(maxtime, maxtime*2, timesteps)
    time_full = np.linspace(0, maxtime*2, timesteps)
    plt.plot(time, observations, label='Observation', color='blue')
    plt.plot(time_next, answers, label='Unseen future', color='green')
    plt.plot(time_full, possible_answers, label='Prediction', color='red')
    plt.title('Pendulum Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    path = config.DIR_FIGURES / "scinet_pendulum_timeserie_prediction.png"
    plt.savefig(path)
    plt.close()    
    return None



if __name__ == "__main__":

    device = config.DEVICE
    kapa_range = config.KAPA_RANGE
    b_range = config.B_RANGE
    timesteps = config.TIMESTEPS
    maxtime = config.MAXTIME

    pendulum_net = PendulumNet(
        input_size=config.M_INPUT_SIZE,
        enc_hidden_sizes=config.M_ENC_HIDDEN_SIZES,
        latent_size=config.M_LATENT_SIZE,
        question_size=config.M_QUESTION_SIZE,
        dec_hidden_sizes=config.M_DEC_HIDDEN_SIZES,
        output_size=config.M_OUTPUT_SIZE
    )
    path = config.DIR_MODEL_PARAMS / f"{config.BEST_MODEL_NAME}.pth"
    pendulum_net.load_state_dict(torch.load(path))


    config.update(SEED=122)  # Change to see different predictions
    from magnetics_diagnostic_analysis.ml_tools.random_seed import seed_everything
    seed_everything(config.SEED)

    N_samples = 1
    observations, questions, answers, params = build_dataset(N_samples, kapa_range, b_range, maxtime=maxtime, timesteps=timesteps)
    possible_answer = make_one_prediction(pendulum_net, observations[0], questions[0], device=device)
    plot_one_prediction(observations[0], questions[0], answers[0], possible_answer, maxtime=maxtime, timesteps=timesteps)
    print("\nPrediction for one random question completed.")


    N_samples = 1
    observations, _, _, params = build_dataset(N_samples, kapa_range, b_range, maxtime=maxtime, timesteps=timesteps)
    answers = data_synthetic_pendulum(params[0][0], params[0][1], t=np.linspace(maxtime, maxtime*2, timesteps))

    questions = np.linspace(0, maxtime*2, timesteps)
    possible_answers = make_timeserie_prediction(pendulum_net, observations[0], questions, device=device)
    plot_timeserie_prediction(observations[0], answers, possible_answers, maxtime=maxtime, timesteps=timesteps)
    print("\nPrediction for full timeserie completed.\n")


