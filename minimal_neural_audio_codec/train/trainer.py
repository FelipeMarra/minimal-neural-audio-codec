import os
from pathlib import Path

from tqdm import tqdm, trange

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

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

        self.writer = SummaryWriter(cfg.tensorboard_path)

    def save_model(self, current_epoch:int, optim:torch.optim.Optimizer):
        save_dict = {
            'epoch': current_epoch -1, # -1 because on the training loop the epoch_idx counts from 1
            'optim_state': self.model.state_dict(),
            'model_state': optim.state_dict()
        }

        save_folder = Path(self.cfg.save_model_path).parent.resolve()
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        torch.save(save_dict, self.cfg.save_model_path)

        return save_folder.absolute()

    def train(self):
        self.model.train()
        self.model = self.model.cuda()
        criterium = nn.MSELoss().cuda()
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        last_eval_loss = float('inf')
        with trange(1, self.cfg.num_epochs+1, desc="Epochs") as epoch_bar:
            for epoch_idx in epoch_bar:
                epoch_cumulative_loss = 0

                train_loader = iter(self.train_dataloader)

                with trange(1, self.cfg.iters_per_epoch+1, desc=f"Epoch {epoch_idx} Iters", leave=False) as batch_bar:
                    for batch_idx in batch_bar:
                        batch = next(train_loader)
                        audios:torch.Tensor = batch['audio']
                        audios = audios.cuda()

                        optim.zero_grad()

                        logits = self.model(audios)
                        loss:torch.Tensor = criterium(logits, audios)
                        loss_item = loss.item()

                        epoch_cumulative_loss += loss_item
                        batch_bar.set_postfix({"current_loss:": loss_item})
                        global_step = batch_idx + self.cfg.iters_per_epoch*(epoch_idx-1)
                        self.writer.add_scalar('Train/Batch Loss', loss_item, global_step)

                        loss.backward()
                        optim.step()

                epoch_train_loss = epoch_cumulative_loss / self.cfg.iters_per_epoch
                epoch_bar.set_postfix({"last_epoch_loss": epoch_train_loss})
                self.writer.add_scalar('Train/Epoch Loss', loss_item, epoch_idx)

                epoch_eval_loss = self.eval()
                tqdm.write(f"Epoch {epoch_idx} train loss: {epoch_train_loss}; eval loss {epoch_eval_loss}")
                self.writer.add_scalar('Eval/Loss', loss_item, epoch_idx)

                if epoch_eval_loss < last_eval_loss:
                    last_eval_loss = epoch_eval_loss
                    save_path = self.save_model(epoch_idx, optim)
                    tqdm.write(f"New best model saved at {save_path}")
        self.writer.flush()

    def eval(self) -> float:
        self.model.eval()

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

        self.model.train()

        eval_loos = cumulative_loss / self.cfg.eval_iters
        return eval_loos
