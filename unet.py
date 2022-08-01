import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=out_channels)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        y = F.relu(self.gn2(self.conv2(x)))
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_block, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, padding=1, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=out_channels)

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))

        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# class Projector(nn.Module):
#     """
#     input: w * h * c
#     output: 128
#     """
#     def __init__(self, in_channels, out_channels=1):
#         super(Projector, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.instance_norm = nn.InstanceNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.dim = 28 * 28
#         self.projector_layer = nn.Sequential(
#             Flatten(),
#             nn.Linear(100, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64)
#             )

#     def forward(self, x):
#         x = self.relu(self.instance_norm(self.conv1(x)))
#         x = self.projector_layer(x)
#         return x

class Projector(nn.Module):
    """
    input: w * h * c
    output: 128
    """
    def __init__(self, in_channels, out_channels=1):
        super(Projector, self).__init__()
        self.projector_layer = nn.Sequential(
            Flatten(),
            nn.Linear(1024*10*10, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            )

    def forward(self, x):
        x = self.projector_layer(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channel=4, out_channel=3):
        super(Unet, self).__init__()
        self.down1 = Downsample_block(in_channel, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_channel, 1)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.gn1(self.conv1(x))))
        x = F.dropout2d(F.relu(self.gn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)
        return x1

class Unet_SimCLR(nn.Module):
    def __init__(self, in_channel=4, out_channel=3):
        super(Unet_SimCLR, self).__init__()
        self.down1 = Downsample_block(in_channel, 64)
        self.down2 = Downsample_block(64, 128)
        self.down3 = Downsample_block(128, 256)
        self.down4 = Downsample_block(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=1024)
        self.up4 = Upsample_block(1024, 512)
        self.up3 = Upsample_block(512, 256)
        self.up2 = Upsample_block(256, 128)
        self.up1 = Upsample_block(128, 64)
        self.outconv = nn.Conv2d(64, out_channel, 1)
        self.projector = Projector(in_channels=1024)

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)
        x = F.dropout2d(F.relu(self.gn1(self.conv1(x))))
        z = self.projector(x)
        x = F.dropout2d(F.relu(self.gn2(self.conv2(x))))
        x = self.up4(x, y4)
        x = self.up3(x, y3)
        x = self.up2(x, y2)
        x = self.up1(x, y1)
        x1 = self.outconv(x)
        return z, x1

if __name__ == "__main__":
    unet = Unet(in_channel=4, out_channel=3)
    x = torch.randn(1, 4, 160, 160)
    y = unet(x)
    print