import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()

        # 480 x 480 x 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 240 x 240 x 16

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 120 x 120 x 32

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 60 x 60 x 64

        self.conv4 = nn.Conv2d(64, 16, (1, 1), padding="same")

        # 60 x 60 x 16

        self.flat = nn.Flatten(start_dim=0)

        self.fc = nn.Sequential(
            nn.Linear(60 * 60 * 16, 256), nn.ReLU(), nn.Linear(256, out_dim)
        )

        # action_dim

    def forward(self, obs):
        x = self.conv1(obs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flat(x)
        x = self.fc(x)
        return F.softmax(x)
