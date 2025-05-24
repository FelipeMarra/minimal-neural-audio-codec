from tqdm import trange

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

#from minimal_neural_audio_codec.config import NeuralAudioCodecConfig #TODO getting import error

class Trainer():
    def __init__(
            self,
            model: nn.Module,
            train_dataset:Dataset, 
            eval_datset:Dataset|None, 
            test_dataset:Dataset|None,
            cfg
        ) -> None:

        self.model = model

        self.train_dataset = train_dataset
        self.eval_datset = eval_datset
        self.test_dataset = test_dataset

        self.batch_size = cfg.data.data_loader.batch_size
        self.num_workers = cfg.data.data_loader.num_workers
        self.num_epochs = cfg.train.epoch.num_epochs
        self.iters_per_epoch = cfg.train.epoch.iters_per_epoch
        self.lr = cfg.train.optim.lr
        self.weight_decay = cfg.train.optim.weight_decay

    def train(self):
        self.model = self.model.cuda()
        criterium = nn.MSELoss().cuda()
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        with trange(1, self.num_epochs+1, desc="Epochs") as epoch_bar:
            for epoch_idx in epoch_bar:
                epoch_cumulative_loss = 0

                train_loader = DataLoader(self.train_dataset, self.batch_size, num_workers=self.num_workers)
                train_loader = iter(train_loader)

                with trange(1, self.iters_per_epoch+1, desc=f"Epoch {epoch_idx} Iters", leave=False) as batch_bar:
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

                epoch_loss = epoch_cumulative_loss / self.iters_per_epoch
                epoch_bar.set_postfix({"last_epoch_loss": epoch_loss})

    def eval(self):
        pass