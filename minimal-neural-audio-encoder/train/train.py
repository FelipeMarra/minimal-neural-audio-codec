from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from nn_modules import NeualAudioCodec

def train(train_loader:DataLoader, modelNeualAudioCodec, epochs:int=100):
    for epoch in epochs:
        pass