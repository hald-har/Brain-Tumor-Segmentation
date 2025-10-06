import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.first(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.second(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x


class DownSample(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class UpSample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class PadAndConcat(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_prev, x_curr):
        diffY = x_prev.size()[2] - x_curr.size()[2]
        diffx = x_prev.size()[3] - x_curr.size()[3]

        x_curr = F.pad(
            x_curr, [diffx // 2, diffx - diffx // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x_prev, x_curr], dim=1)
        return x


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        self.down_conv = nn.ModuleList(
            [
                DoubleConv(i, o)
                for i, o in [
                    (in_channels, features),
                    (features, 2 * features),
                    (2 * features, 4 * features),
                    (4 * features, 8 * features),
                ]
            ]
        )  # downsampling convolution blocks

        self.down_sample = nn.ModuleList(
            [DownSample() for _ in range(len(self.down_conv))]
        )  # downsampling layers

        self.bottleneck = DoubleConv(8 * features, 16 * features)  # bottleneck layer

        self.up_sample = nn.ModuleList(
            [
                UpSample(i, o)
                for i, o in [
                    (16 * features, 8 * features),
                    (8 * features, 4 * features),
                    (4 * features, 2 * features),
                    (2 * features, features),
                ]
            ]
        )  # upsampling layers

        self.up_conv = nn.ModuleList(
            [
                DoubleConv(i, o)
                for i, o in [
                    (16 * features, 8 * features),
                    (8 * features, 4 * features),
                    (4 * features, 2 * features),
                    (2 * features, features),
                ]
            ]
        )  # upsampling convolution blocks

        self.pad_and_concat = nn.ModuleList(
            [PadAndConcat() for _ in range(len(self.up_conv))]
        )  # padding and concatenation layers

        self.final_conv = nn.Conv2d(
            features, out_channels, kernel_size=1
        )  # final output layer

    def forward(self, x):
        skip_connections = []

        for i in range(len(self.down_conv)):
            x = cp.checkpoint(self.down_conv[i], x ,use_reentrant=True)
            skip_connections.append(x)
            x = self.down_sample[i](x)

        x = cp.checkpoint(self.bottleneck, x , use_reentrant=True)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.pad_and_concat[i](x, skip_connections[i])
            x = cp.checkpoint(self.up_conv[i], x ,use_reentrant=True)

        x = self.final_conv(x)
        return x
