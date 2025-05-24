from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import DataLoader

from train.train_config import TrainConfig

class Trainer():
    def __init__(
            self,
            model: nn.Module,
            train_dataloader:DataLoader, 
            eval_dataloader:DataLoader, 
            test_dataloader:DataLoader,
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

                epoch_train_loss = epoch_cumulative_loss / self.cfg.iters_per_epoch
                epoch_bar.set_postfix({"last_epoch_loss": epoch_train_loss})

                epoch_eval_loss = self.eval()

                tqdm.write(f"Epoch {epoch_idx} train loss: {epoch_train_loss}; eval loss {epoch_eval_loss}")

    def eval(self) -> float:
        cumulative_loss = 0

        with torch.no_grad():
            eval_loader = iter(self.eval_dataloader)
            criterium = nn.MSELoss().cuda()

            with trange(1, self.cfg.eval_iters+1, desc="Eval", leave=False) as eval_bar:
                for _ in eval_bar:
                    batch = next(eval_loader)
                    audios:torch.Tensor = batch['audio']
                    audios = audios.cuda()

                    logits = self.model(audios)
                    loss:torch.Tensor = criterium(logits, audios)
                    cumulative_loss += loss.item()

        eval_loos = cumulative_loss / self.cfg.eval_iters
        return eval_loos
