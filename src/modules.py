from itertools import accumulate
import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pool(x)


def build_encoder(channels):
    return nn.ModuleList([
        DoubleConv(cin, cout) for cin, cout in zip(channels, channels[1:])
    ])

def build_decoder(channels):
    return nn.ModuleList([
        nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        for cin, cout in zip(channels, channels[1:])
    ])


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, c=64):
        super().__init__()
        
        # Time embeddings for each skip connection
        self.time_mlp1 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, c),
            nn.SiLU(),
            nn.Linear(c, c)
        )
        
        self.time_mlp2 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 2*c),
            nn.SiLU(),
            nn.Linear(2*c, 2*c)
        )
        
        self.time_mlp3 = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, 4*c),
            nn.SiLU(),
            nn.Linear(4*c, 4*c)
        )
        
        # Encoder
        self.downconvs = build_encoder([in_channels, c, 2*c, 4*c])
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(3)])
        self.bottleneck = DoubleConv(4*c, 8*c)

        # Decoder
        self.upconvs = build_decoder([8*c, 4*c, 2*c, c])
        self.dec_convs = nn.ModuleList([
            DoubleConv(4*c + 4*c + 4*c, 4*c),
            DoubleConv(2*c + 2*c + 2*c, 2*c),
            DoubleConv(c + c + c, c)
        ])
        
        self.final_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    
    def forward(self, x, t):
        # Time embeddings
        t1 = self.time_mlp1(t)
        t2 = self.time_mlp2(t)
        t3 = self.time_mlp3(t)
        
        skips = []
        # Encoder
        for downconv, pool in zip(self.downconvs, self.pools):
            x = downconv(x)
            skips.append(x)
            x = pool(x)
        
        x = self.bottleneck(x)
        
        # Decoder
        skips = skips[::-1]
        time_embs = [t3, t2, t1]

        for i, (upconv, dec_conv, skip) in enumerate(zip(self.upconvs, self.dec_convs, skips)):
            x = upconv(x)
            
            if x.shape != skip.shape:
                # Pad to match spatial dimensions
                diffY = skip.shape[2] - x.shape[2]
                diffX = skip.shape[3] - x.shape[3]
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            time_emb = time_embs[i]
            t_emb = time_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])
            x = torch.cat((skip, x, t_emb), dim=1)
            x = dec_conv(x)

        return self.final_conv(x)
        