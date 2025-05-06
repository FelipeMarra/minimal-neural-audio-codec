import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUnit(nn.Module):
    # TODO: bias false? activation function inplace? this much batch norms? use bottle neck design?
    # https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5
    def __init__(self, in_channels:int, out_channels:int):
        """
            Args:
                in_channels: number of input channels in the first layer
        """
        super(ResidualUnit, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False), # TODO: should be equivalent to a linear linear, so might be beter to use one
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ELU()(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = nn.ELU()(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stride:int):
        super(ConvBlock, self).__init__()

        self.res_unit = ResidualUnit(in_channels, out_channels)
        self.downsample = nn.Conv1d(out_channels, out_channels, stride*2, stride, bias=False) # kernel size is twice the stride

    def forward(self, x):
        x = self.res_unit(x)
        x = self.downsample(x) # TODO no activation?

        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # EnCodec has 150 lattent steps per second for 48kHz
        # We will get 130 lattent steps for 44.1 kHz

        # They say they use 32 channels and kernel of size 7
        self.conv1 = nn.Conv1d(2, 32, 7)

        # Number of channels is doubled whenever downsample occurs
        self.conv_block_1 = ConvBlock(32, 64, 2) 
        self.conv_block_2 = ConvBlock(64, 128, 4)
        self.conv_block_3 = ConvBlock(128, 256, 5)
        self.conv_block_4 = ConvBlock(256, 512, 8)

        self.lstm = nn.LSTM(
            input_size=1, # we'll go over our vallues one by one for each batch, for each channel
            hidden_size=64, # random choice, since they dont specify it
            proj_size=1, # we need to go back to a single number 
            num_layers=2,
            batch_first=True # we pass data in (batch, seq, feature) format
        )

        self.conv2 = nn.Conv1d(512, 1024, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = F.elu(x) # [2, 32, 44094]

        x:torch.Tensor = self.conv_block_1(x) # [2, 64, 22046] -> 44.094 -3 do kernel e /2 do stride -> 22,045.5
        x:torch.Tensor = self.conv_block_2(x) # [2, 128, 5510]
        x:torch.Tensor = self.conv_block_3(x) # [2, 256, 1101]

        x:torch.Tensor = self.conv_block_4(x) # [2, 512, 136]

        B, C, S = x.shape # (batch, channel, sequence) -> there is a seq for each batch for each channel
        x = x.reshape(B*C, S, 1) # (batch, seq, feature) -> we need to treat every channel as a batch to have the seq len in the second dim and the feature in the third

        x, (_, _) = self.lstm(x) # out, (h, c) [1024, 136, 1]

        x = x.reshape(B, C, S) # restore shape to (batch, channel, sequence)

        x = self.conv2(x) # [2, 1024, 130]

        print(x.shape)