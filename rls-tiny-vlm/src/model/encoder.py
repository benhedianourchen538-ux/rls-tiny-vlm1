"""CNN vision encoder (no pretrained weights): image (B,3,64,64) -> feature map (B,C,H',W').

Output size of a conv/pool layer:  out = floor((n + 2*padding - kernel) / stride) + 1
  * 3x3 conv, stride 1, padding 1  -> size unchanged
  * 2x2 max-pool, stride 2         -> size halved
So with 3 poolings: 64 -> 32 -> 16 -> 8, i.e. an 8x8 grid = 64 "visual tokens".
Pooling discards exact position inside each 2x2 cell; position is recovered later from the
positional embeddings that the decoder adds to the visual tokens.
"""
import torch
import torch.nn as nn


def conv_out_size(n: int, kernel: int, stride: int = 1, padding: int = 0) -> int:
    return (n + 2 * padding - kernel) // stride + 1


class ConvBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int):
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class CNNEncoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 128), n_pools: int = 3, in_ch: int = 3):
        super().__init__()
        layers, c_in = [], in_ch
        for i, c in enumerate(channels):
            layers.append(ConvBlock(c_in, c))
            if i < n_pools:
                layers.append(nn.MaxPool2d(2))
            c_in = c
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        self.n_pools = n_pools

    def out_size(self, img_size: int = 64) -> int:
        n = img_size
        for _ in range(self.n_pools):
            n = conv_out_size(n, kernel=2, stride=2)
        return n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:                       # (B,3,64,64) uint8 -> float in [-1, 1]
            x = x.float() / 127.5 - 1.0
        return self.net(x)                               # (B, C, H', W')
