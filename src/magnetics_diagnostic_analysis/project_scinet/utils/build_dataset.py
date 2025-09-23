import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import random_split

from magnetics_diagnostic_analysis.project_scinet.utils.data_creation_pendulum import data_synthetic_pendulum


def build_dataset(num_samples=1000, kapa_range=(3.0, 8.0), b_range=(0.1, 1.0), maxtime=5.0, timesteps=50):
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
        question = np.random.uniform(timesteps, timesteps*2)
        questions.append(question)
        # Build answer to the question
        a_corr.append(data_synthetic_pendulum(kapa, b, t=np.array(question), maxtime=maxtime, timesteps=50))
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

    N_samples = 10000
    kapa_range = (5.0, 6.0)
    b_range = (0.2, 0.5)
    observations, questions, answers, params = build_dataset(N_samples, kapa_range, b_range)

    dataset = PendulumDataset(observations, questions, answers, params)


    # Split into training and validation sets

    train_valid_rate = 0.8
    train_size = int(train_valid_rate * len(dataset))
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])







