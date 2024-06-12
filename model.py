import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, mode1, mode2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode1 = (
            mode1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.mode2 = mode2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.mode1, self.mode2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.mode1, self.mode2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.mode1, : self.mode2] = self.compl_mul2d(
            x_ft[:, :, : self.mode1, : self.mode2], self.weights1
        )
        out_ft[:, :, -self.mode1 :, : self.mode2] = self.compl_mul2d(
            x_ft[:, :, -self.mode1 :, : self.mode2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.norm = nn.GroupNorm(num_groups=2, num_channels=mid_channels)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, mode1, mode2, width, L=10):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.mode1 = mode1
        self.mode2 = mode2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic
        self.fourier_depth = 4
        self.L = L

        self.p = nn.Linear(
            3 + self.L * 4, self.width
        )  # input channel is 43: (a(x, y), x, y) + L * 2 (x, y) * 2 (sin,cos)
        self.convs = nn.ModuleList(
            [
                SpectralConv2d(self.width, self.width, self.mode1, self.mode2)
                for _ in range(self.fourier_depth)
            ]
        )
        self.mlps = nn.ModuleList(
            [MLP(self.width, self.width, self.width) for _ in range(self.fourier_depth)]
        )
        self.ws = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.fourier_depth)]
        )
        self.norms = nn.ModuleList(
            [
                nn.GroupNorm(num_groups=2, num_channels=self.width)
                for _ in range(self.fourier_depth)
            ]
        )
        self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)  # (batchsize, 85, 85, 2)
        pos_encoding = self.positional_encoding(grid)  # (batchsize, 85, 85, 40)
        x = torch.cat((x, grid, pos_encoding), dim=-1)

        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.fourier_depth):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x2 = self.norms[i](x2)
            x = x1 + x2
            if i != self.fourier_depth - 1:
                x = F.gelu(x)

        x = x[..., : -self.padding, : -self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def positional_encoding(self, x, cst=torch.pi):
        """
        Positional encoding for the input data.
        """
        cos_encoding = []
        sin_encoding = []

        for pos in range(self.L):
            freq = cst * (2**pos)
            cos_encoding.append(torch.cos(freq * x))
            sin_encoding.append(torch.sin(freq * x))

        # Concatenate along the last dimension
        cos_encoding = torch.cat(cos_encoding, dim=-1)
        sin_encoding = torch.cat(sin_encoding, dim=-1)

        # Concatenate cos and sin encodings along the last dimension
        encoding = torch.cat((cos_encoding, sin_encoding), dim=-1)

        return encoding

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

        return torch.cat((gridx, gridy), dim=-1).to(device)
