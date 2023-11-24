import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet50_Weights,
)


class Resnet2D(nn.Module):
    """
    Pretrained Resnet50 (2D only)
    """

    def __init__(self, in_channels: int, out_size: int, resnet_type: str = "resnet18"):
        super(Resnet2D, self).__init__()

        if resnet_type == "resnet18":
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif resnet_type == "resnet50":
            self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.resnet.requires_grad_(False)

        # fit first layer for our input

        if in_channels > 3:
            new_conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.resnet.conv1.out_channels,
                kernel_size=self.resnet.conv1.kernel_size,
                stride=self.resnet.conv1.stride,
                padding=self.resnet.conv1.padding,
                bias=self.resnet.conv1.bias,
            )

            copy_weights = 0  # idx of existing weights to initialize new channel weights with (this is red channel)
            new_conv1.weight[
                :, : self.resnet.conv1.in_channels, :, :
            ] = self.resnet.conv1.weight.clone()

            for i in range(in_channels - self.resnet.conv1.in_channels):
                channel = self.resnet.conv1.in_channels + i
                new_conv1.weight[
                    :, channel : channel + 1, :, :
                ] = self.resnet.conv1.weight[:, copy_weights, :, :].clone()

            new_conv1.weight = nn.Parameter(new_conv1.weight)

            self.resnet.conv1 = new_conv1

        # fit output size
        self.resnet.fc = nn.Linear(
            in_features=self.resnet.fc.in_features, out_features=out_size
        )

    def forward(self, x: torch.Tensor):
        x = x.type(torch.cuda.FloatTensor)
        return self.resnet(x)
