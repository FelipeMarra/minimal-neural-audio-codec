from enum import Enum

import torch

from nn_modules.decoder.decoder import Decoder
from nn_modules.encoder.encoder import Encoder

class CodecMode(Enum):
    TRAIN = 0
    ENCODE = 1
    DECODE = 2

class NeuralAudioCodec(torch.nn.Module):
    def __init__(self, mode:CodecMode=CodecMode.TRAIN, in_t=44100, verbose=False):
        super(NeuralAudioCodec, self).__init__()

        self.mode = mode

        self.encoder = Encoder(in_t=in_t, verbose=verbose)
        self.decoder = Decoder(in_t=137, in_c=1024, verbose=verbose)

    def forward(self, x):
        if self.mode in [CodecMode.TRAIN, CodecMode.ENCODE]:
            x = self.encoder(x)

        if self.mode in [CodecMode.TRAIN, CodecMode.DECODE]:
            x = self.decoder(x)

        return x
