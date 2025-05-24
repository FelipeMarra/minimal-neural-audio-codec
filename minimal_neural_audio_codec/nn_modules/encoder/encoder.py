from utils import get_padding, get_conv_out

import torch
import torch.nn as nn

class Conv1dLN(nn.Module):    
    def __init__(self, in_t:int, in_channels:int, out_channels:int, kernel:int=3, stride:int=1, padding:int=0, bias:bool=True):
        """
        We use layer normalization (Ba et al., 2016), computing the statistics including also the time dimension in order to keep the relative scale information.

        Args:
            in_channels: number of input channels in the first layer
            out_channels: number of channels to output
            padding: padding to apply on both sides
            in_t: dimention of the last dim, that is, the time dim
        """
        super(Conv1dLN, self).__init__()
        self.out_channels = out_channels

        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride, padding, bias=bias)
        self.out_t = get_conv_out(in_t, kernel, stride, padding)
        self.ln1 = nn.LayerNorm((out_channels, self.out_t)) # type: ignore

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.ln1(x)

class ResidualUnit(nn.Module):
    # TODO: bias false? activation function inplace? this much batch norms? use bottle neck design?
    # https://wandb.ai/amanarora/Written-Reports/reports/Understanding-ResNets-A-Deep-Dive-into-Residual-Networks-with-PyTorch--Vmlldzo1MDAxMTk5
    def __init__(self, in_t:int, in_channels:int, out_channels:int, kernel:int=3, padding:int=0):
        """
        Args:
            in_channels: number of input channels in the first layer
            out_channels: number of channels to output
            padding: padding to apply on both sides
            in_t: dimention of the last dim, that is, the time dim
        """
        super(ResidualUnit, self).__init__()

        self.out_channels = out_channels

        self.conv1 = Conv1dLN(in_t, in_channels, out_channels, kernel, padding=1)

        #conv2_padding = self.get_res_unit_conv2_padding(in_t, kernel) TODO: ignoring because its getting too complicated
        self.conv2 = Conv1dLN(self.conv1.out_t, out_channels, out_channels, kernel, padding=1)

        self.out_t = self.conv2.out_t

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = Conv1dLN(in_t, in_channels, out_channels, kernel, padding=1, bias=False)

    def get_res_unit_conv2_padding(self, in_t, kernel, stride=1):
        """
            To get the value of the padding of the second convolution in order to
            make its output the same as the one of the shortcut layer

            This was basically obtained by doing 
                `get_conv_out(get_conv_out(in_t)) = get_conv_out(in_t)`
            and isolating the padding
        """
        if stride == 1:
            return (kernel-1)//2

        return ((in_t-1)*stride+kernel-in_t)//2

    def forward(self, x) -> torch.Tensor:
        out = self.conv1(x)
        out = nn.ELU()(out)

        out = self.conv2(out)

        out += self.shortcut(x)
        out = nn.ELU()(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_t:int, in_channels:int, out_channels:int, stride:int):
        """
        Args:
            in_channels: number of input channels in the first layer
            out_channels: number of channels to output
            stride: stride to apply on the downsample conv
            in_t: dimention of the last dim, that is, the time dim
        """
        super(ConvBlock, self).__init__()

        self.out_channels = out_channels
        kernel = stride*2 # kernel size is twice the stride
        padding = get_padding(kernel, stride)

        self.res_unit = ResidualUnit(in_t, in_channels, out_channels, padding=padding)
        self.downsample = Conv1dLN(self.res_unit.out_t, out_channels, out_channels, kernel, stride, padding, bias=False)

        self.out_t = self.downsample.out_t

    def forward(self, x) -> torch.Tensor:
        x = self.res_unit(x)
        x = self.downsample(x)
        x = nn.ELU()(x)

        return x

class Encoder(nn.Module):
    def __init__(self, in_t:int=44100, verbose:bool=False):
        """
        Args:
            in_t: dimention of the last dim, that is, the time dim. Also known as the sample rate.
        """
        super(Encoder, self).__init__()

        kernel = 7
        out_channels = 32
        padding = get_padding(kernel)
        self.verbose = verbose

        self.conv1 = Conv1dLN(in_t, 2, out_channels, kernel, padding=padding)

        self.conv_block_1 = ConvBlock(
            in_t = self.conv1.out_t,
            in_channels = self.conv1.out_channels, # 32
            out_channels = 2*self.conv1.out_channels, # 64 Number of channels is doubled whenever downsample occurs
            stride = 2
        )

        self.conv_block_2 = ConvBlock(
            in_t = self.conv_block_1.out_t,
            in_channels = self.conv_block_1.out_channels, # 64
            out_channels = 2*self.conv_block_1.out_channels, # 128
            stride = 4
        )

        self.conv_block_3 = ConvBlock(
            in_t = self.conv_block_2.out_t,
            in_channels = self.conv_block_2.out_channels, # 128
            out_channels = 2*self.conv_block_2.out_channels, # 256
            stride = 5
        )

        self.conv_block_4 = ConvBlock(
            in_t = self.conv_block_3.out_t,
            in_channels = self.conv_block_3.out_channels, # 256
            out_channels = 2*self.conv_block_3.out_channels, # 512
            stride = 8
        )

        # TODO Looks like somehow their lstm is like LSTM(1024, 1024, num_layers=2)
        # My logic is that we have a batch of dimention B, C channels and sequence of size S
        # We need the LSTM to go over every sequence to model time, so it would go over the
        # last dim of a tensor of shape (B*C, S, 1) since the LSTM expects (batch, seq, feature)
        # when batch_first is true
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            proj_size=1, # we need to go back to a single number 
            num_layers=2,
            batch_first=True # we pass data in (batch, seq, feature) format
        )

        # final 1D convolution layer with a kernel size of 7 and D output channels
        # What value should I use for D? I'll double the prevoius out channels, since it has been the pattern so far
        self.conv2 = Conv1dLN(
            in_t = self.conv_block_4.out_t,
            in_channels = self.conv_block_4.out_channels, # 512
            out_channels = 2*self.conv_block_4.out_channels, # 1024
            kernel = kernel, 
            padding=padding
        )

    def forward(self, x): # type: ignore
        # x is [2, 2, 44100]
        x = self.conv1(x)
        x = nn.ELU()(x) # [2, 32, 44100]
        if self.verbose: print(f"Encoder Conv 1 {x.shape}")

        x:torch.Tensor = self.conv_block_1(x) # [2, 64, 22050]
        if self.verbose: print(f"Encoder Block 1 {x.shape}")
        x:torch.Tensor = self.conv_block_2(x) # [2, 128, 5512]
        if self.verbose: print(f"Encoder Block 2 {x.shape}")
        x:torch.Tensor = self.conv_block_3(x) # [2, 256, 1102]
        if self.verbose: print(f"Encoder Block 3 {x.shape}")
        x:torch.Tensor = self.conv_block_4(x) # [2, 512, 137]
        if self.verbose: print(f"Encoder Block 4 {x.shape}")

        # (batch, channel, sequence) -> there is a seq for each batch for each channel
        B, C, S = x.shape

        # (batch, seq, feature) -> we need to treat every channel as a batch
        # to have the seq len in the second dim and the feature in the third
        x = x.reshape(B*C, S, 1) 

        x, (_, _) = self.lstm(x) # out, (h, c) [1024, 137, 1]

        x = x.reshape(B, C, S) # restore shape to (batch, channel, sequence)
        if self.verbose: print(f"Encoder LSTM {x.shape}")

        x = self.conv2(x) # [2, 1024, 137]
        if self.verbose: print(f"Encoder Conv 2 {x.shape}\n")

        return x