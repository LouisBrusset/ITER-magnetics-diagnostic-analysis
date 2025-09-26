import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import random_split

from magnetics_diagnostic_analysis.project_scinet.setting_scinet import config
from magnetics_diagnostic_analysis.project_scinet.utils.data_creation_pendulum import data_synthetic_pendulum




def build_dataset(num_samples=1000, kapa_range=(3.0, 8.0), b_range=(0.1, 1.0), maxtime=5.0, timesteps=500):
    observations = []
    questions = []
    params = []
    a_corr = []

    for _ in range(num_samples):
        # Build observations
        kapa = np.random.uniform(*kapa_range)
        b = np.random.uniform(*b_range)
        params.append((kapa, b))
        timeserie = data_synthetic_pendulum(kapa, b, timesteps=timesteps, maxtime=maxtime)
        observations.append(timeserie)

        # Build questions
        question = np.random.uniform(0, maxtime*2)
        questions.append(question)

        # Build answer to the question
        a_corr.append(data_synthetic_pendulum(kapa, b, t=np.array(question)))
    
    return np.array(observations), np.array(questions), np.array(a_corr), np.array(params)


class PendulumDataset(Dataset):
    def __init__(self, observations, questions, answers, params):
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.questions = torch.tensor(questions, dtype=torch.float32)
        self.answers = torch.tensor(answers, dtype=torch.float32)
        self.params = torch.tensor(params, dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.questions[idx], self.answers[idx], self.params[idx]
    




if __name__ == "__main__":

    # Create dataset

    N_samples = config.N_SAMPLES
    kapa_range = config.KAPA_RANGE
    b_range = config.B_RANGE
    timesteps = config.TIMESTEPS
    maxtime = config.MAXTIME

    observations, questions, answers, params = build_dataset(N_samples, kapa_range, b_range, maxtime=maxtime, timesteps=timesteps)
    dataset = PendulumDataset(observations, questions, answers, params)
    print("\nCreation of dataset completed.\n")


    # Split into training and validation sets

    train_valid_rate = config.TRAIN_VALID_SPLIT
    train_size = int(train_valid_rate * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print("\nSplit into training and validation sets completed.\n")

    # Save datasets
    path_train = config.DIR_SYNTHETIC_DATA / "pendulum_scinet_train.pt"
    path_valid = config.DIR_SYNTHETIC_DATA / "pendulum_scinet_valid.pt"
    torch.save(train_dataset, path_train)
    torch.save(valid_dataset, path_valid)

    print(f"Training dataset saved at: {path_train}")
    print(f"Validation dataset saved at: {path_valid}\n")






