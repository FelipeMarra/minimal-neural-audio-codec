from dataclasses import dataclass

# Data
@dataclass
class Dataset:
    path: str

@dataclass
class Dataloader:
    batch_size: int
    num_workers: int

## Data agregator
@dataclass
class Data:
    dataset: Dataset
    data_loader: Dataloader

# Train
@dataclass
class Epoch:
    num_epochs: int
    iters_per_epoch: int

@dataclass
class Optim:
    lr: float
    weight_decay: float

## Train agregator
@dataclass
class Train:
    epoch: Epoch
    optim: Optim

## Model
@dataclass
class Model:
    verbose: bool

# Agregating the configs dataclasses
@dataclass
class NeuralAudioCodecConfig:
    data: Data
    train: Train
    model: Model