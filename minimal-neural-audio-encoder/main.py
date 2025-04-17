#%%
SNES_ROOT = "/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/snes-back/vmdb/nintendo-snes-spc"
from datasets.senes import SNESDataset

#%%
snes_dataset = SNESDataset(SNES_ROOT)

# %%
print(len(snes_dataset))
print(snes_dataset[0]['audio'].size())
print(snes_dataset[0]['audio'][0][0])