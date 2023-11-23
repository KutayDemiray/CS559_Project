# https://github.com/jan-xu/autoencoders/blob/master/resnet/resnet.ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, mode="encode"):
        super(ResBlock, self).__init__()

        if mode == "encode":
            self.conv1 = nn.Conv2d(c_in, c_out, kernel_size, stride, padding)
            self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        elif mode == "decode":
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding)
            self.conv2 = nn.ConvTranspose2d(
                c_out, c_out, kernel_size=3, stride=1, padding=1
            )

        self.relu = nn.ReLU()
        # self.batch_norm = nn.BatchNorm2d(c_out)

        self.resize = stride > 1 or (stride == 1 and padding == 0) or c_out != c_in

    def forward(self, x):
        identity = x.clone()
        z = self.conv1(x)
        # z = F.batch_norm(z)
        z = F.relu(z)
        z = self.conv2(z)
        # z = F.batch_norm(z)
        if self.resize:
            identity = self.conv1(identity)
        # y = F.batch_norm(y)
        return F.relu(identity + z)


class Encoder(nn.Module):
    def __init__(self, in_channels: int):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1
        )  # 16,
        # self.batch_norm = nn.BatchNorm2d(16)

        self.rb1 = ResBlock(c_in=16, c_out=16, kernel_size=3, stride=2, padding=1)
        self.rb2 = ResBlock(c_in=16, c_out=32, kernel_size=3, stride=1, padding=1)
        self.rb3 = ResBlock(c_in=32, c_out=32, kernel_size=3, stride=2, padding=1)
        self.rb4 = ResBlock(c_in=32, c_out=48, kernel_size=3, stride=1, padding=1)
        self.rb5 = ResBlock(c_in=48, c_out=48, kernel_size=3, stride=2, padding=1)
        self.rb6 = ResBlock(c_in=48, c_out=64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: 3 x 480 x 480
        z = self.conv1(x)
        # z = F.batch_norm(z)
        z = F.relu(z)
        z = self.rb1(z)  #
        z = self.rb2(z)
        z = self.rb3(z)
        z = self.rb4(z)
        z = self.rb5(z)
        z = self.rb6(z)
        return z


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.rb1 = ResBlock(64, 48, 2, 2, 0, "decode")  # 48 4 4
        self.rb2 = ResBlock(48, 48, 2, 2, 0, "decode")  # 48 8 8
        self.rb3 = ResBlock(48, 32, 3, 1, 1, "decode")  # 32 8 8
        self.rb4 = ResBlock(32, 32, 2, 2, 0, "decode")  # 32 16 16
        self.rb5 = ResBlock(32, 16, 3, 1, 1, "decode")  # 16 16 16
        self.rb6 = ResBlock(16, 16, 2, 2, 0, "decode")  # 16 32 32
        self.out_conv = nn.ConvTranspose2d(16, out_channels, 3, 1, 1)  # 3 32 32
        self.tanh = nn.Tanh()

    def forward(self, x):
        z = self.rb1(x)
        z = self.rb2(z)
        z = self.rb3(z)
        z = self.rb4(z)
        z = self.rb5(z)
        z = self.rb6(z)
        z = self.out_conv(z)
        output = F.tanh(z)
        return output


class ResnetAutoencoder(nn.Module):
    def __init__(self, in_channels: int):
        super(ResnetAutoencoder, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(in_channels)

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat
