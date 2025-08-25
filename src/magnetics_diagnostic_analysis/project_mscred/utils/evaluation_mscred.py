import numpy as np
import matplotlib.pyplot as plt
import torch

from magnetics_diagnostic_analysis.project_mscred.setting_mscred import config
from magnetics_diagnostic_analysis.project_mscred.utils.dataloader_building import create_data_loaders










if __name__ == "__main__":

    data = np.random.rand(2000, *config.DATA_SHAPE)

    _, valid_loader, test_loader = create_data_loaders(data)

    






















