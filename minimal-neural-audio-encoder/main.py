#%%
from torch.utils.data import DataLoader
from datasets.senes import SNESDataset
from nn_modules.encoder.encoder import Encoder
from nn_modules.decoder.decoder import Decoder

SNES_ROOT = "/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/snes-back/vmdb/nintendo-snes-spc"

#%%
snes_dataset = SNESDataset(SNES_ROOT)

# %%
print(len(snes_dataset))
print(snes_dataset[0]['path'])
print(snes_dataset[0]['sr'])
print(snes_dataset[0]['audio'].shape)
print(snes_dataset[0]['audio'][0][0])
print()

# %%
snes_loader = DataLoader(snes_dataset, 2)
batch = next(iter(snes_loader))
print('Batch Audio Shape:', batch['audio'].shape)

#%%
out = Encoder()(batch['audio']) 
#out = Decoder()(out)

# %%
#TODO
# We further split the input into chunks of 1 seconds, with an overlap of 10 ms to avoid clicks, and normalize each
# chunk before feeding it to the model, applying the inverse operation on the output of the decoder, adding a
# negligible bandwidth overhead to transmit the scale.
# https://discuss.pytorch.org/t/how-to-normalize-audio-data-in-pytorch/187709