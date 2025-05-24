from enum import Enum
from dataclasses import dataclass

class SplitEnum(Enum):
    TRAIN = 'train'
    EVAL = 'eval'
    TEST = 'test'

@dataclass
class DatasetConfig:
    games_genres: str
    path: str

@dataclass
class DataloaderConfig:
    batch_size: int
    num_workers: int

@dataclass
class DataConfig:
    dataset: DatasetConfig
    data_loader: DataloaderConfig