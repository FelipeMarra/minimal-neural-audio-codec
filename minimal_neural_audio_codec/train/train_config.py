from dataclasses import dataclass

@dataclass
class TrainConfig:
    num_epochs: int
    iters_per_epoch: int
    lr: float
    weight_decay: float