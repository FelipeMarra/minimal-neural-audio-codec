from tqdm import trange

import torch
from torch import nn
from torch.utils.data import DataLoader

from train.train_config import TrainConfig

class Trainer():
    def __init__(
            self,
            model: nn.Module,
            train_dataloader:DataLoader, 
            eval_dataloader:DataLoader|None, 
            test_dataloader:DataLoader|None,
            cfg: TrainConfig
        ) -> None:

        self.model = model

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        self.cfg = cfg

    def train(self):
        self.model = self.model.cuda()
        criterium = nn.MSELoss().cuda()
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        with trange(1, self.cfg.num_epochs+1, desc="Epochs") as epoch_bar:
            for epoch_idx in epoch_bar:
                epoch_cumulative_loss = 0

                train_loader = iter(self.train_dataloader)

                with trange(1, self.cfg.iters_per_epoch+1, desc=f"Epoch {epoch_idx} Iters", leave=False) as batch_bar:
                    for _ in batch_bar:
                        batch = next(train_loader)
                        audios:torch.Tensor = batch['audio']
                        audios = audios.cuda()

                        optim.zero_grad()

                        logits = self.model(audios)
                        loss:torch.Tensor = criterium(logits, audios)

                        epoch_cumulative_loss += loss.item() 
                        batch_bar.set_postfix({"current_loss:": loss.item()})

                        loss.backward()
                        optim.step()

                epoch_loss = epoch_cumulative_loss / self.cfg.iters_per_epoch
                epoch_bar.set_postfix({"last_epoch_loss": epoch_loss})

    def eval(self):
        pass