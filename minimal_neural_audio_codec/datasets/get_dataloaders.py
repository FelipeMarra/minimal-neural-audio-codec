from datasets import DataConfig, SNESDataset 
from datasets.datasets_config import SplitEnum

from torch.utils.data import DataLoader

def get_dataloader(cfg: DataConfig, mode:SplitEnum):
    loader = DataLoader(
        dataset=SNESDataset(cfg.dataset.path, mode), 
        batch_size=cfg.data_loader.batch_size, 
        num_workers=cfg.data_loader.num_workers
    )

    return loader