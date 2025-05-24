from dataclasses import dataclass
from train.train_config import TrainConfig
from datasets.datasets_config import DataConfig

## Model
@dataclass
class ModelConfig:
    verbose: bool

# Agregating the configs dataclasses
@dataclass
class NeuralAudioCodecConfig:
    data: DataConfig
    train: TrainConfig
    model: ModelConfig