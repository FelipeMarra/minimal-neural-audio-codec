import torch.nn as nn

class Encoder(nn.Modoule):
    def __init__(self):
        super(Encoder, self).__init__()

        # EnCodec have 150 lattent steps per second for 48kHz
        # Our audios are at 44.1 kHz, but I think 150 is a fine number
        # So we need to cover 294 data points per step to get 150 lattent steps?
        self.conv1 = nn.Conv1d(2, 32, 7) # They say they use 32 channels and kernel of size 7
