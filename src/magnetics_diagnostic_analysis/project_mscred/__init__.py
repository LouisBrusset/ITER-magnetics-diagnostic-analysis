"""
MSCRED Project Module

Multi-Scale Convolutional Recurrent Encoder Decoder for anomaly detection.
"""

from .utils.matrix_generator import generate_signature_matrix
from .utils.dataloader_building import create_data_loaders
from .utils.synthetic_anomaly_adding import create_anomalies
from .utils.mast_data_scraping import *

from .model.convlstm import ConvLSTMCell, ConvLSTM
from .model.mscred import attention, CnnEncoder, CnnDecoder, Conv_LSTM, MSCRED

__all__ = [
    "generate_signature_matrix",
    "create_data_loaders",
    "create_anomalies",
    "ConvLSTMCell",
    "ConvLSTM",
    "attention",
    "CnnEncoder",
    "CnnDecoder",
    "Conv_LSTM",
    "MSCRED"
]