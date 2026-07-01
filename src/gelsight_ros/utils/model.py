import torch
import torch.nn as nn
import torch.nn.functional as F

class AnglePredictor(nn.Module):
    # def __init__(self):
    #     super(AnglePredictor, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
    #     nn.Conv2d()
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
    #     self.fc1 = nn.Linear(32 * 53 * 53, 128)
    #     self.fc2 = nn.Linear(128, 1)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 32 * 53 * 53)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x
    """
    Input:  (B, 3, 480, 640)  float32
    Output: (B, 2)            float32
    """
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            # 3 x 480 x 640
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> 32 x 240 x 320
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64 x 120 x 160
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 128 x 60 x 80
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # -> 256 x 30 x 40
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # -> 512 x 15 x 20
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # collapse spatial dims without caring about exact WxH
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 512 x 1 x 1
        )

        self.head = nn.Sequential(
            nn.Flatten(),                 # -> 512
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),            # -> 2 floats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class SimpleModel(nn.Module):
    """
    Moderate simplification of AnglePredictor.
    Reduces depth: 5 conv layers -> 3 conv layers
    Increases regularization: dropout 0.5, L2 via weight_decay
    
    Input:  (B, 3, 480, 640)  float32
    Output: (B, 2)            float32
    """
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            # 3 x 480 x 640
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> 32 x 240 x 320
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64 x 120 x 160
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 128 x 60 x 80
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # collapse spatial dims
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 128 x 1 x 1
        )

        self.head = nn.Sequential(
            nn.Flatten(),                 # -> 128
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2),            # -> 2 floats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class SimplerModel(nn.Module):
    """
    Aggressive simplification of AnglePredictor.
    Reduces depth: 5 conv layers -> 2 conv layers
    Minimal capacity, heavy dropout (0.5+) for maximum regularization
    
    Input:  (B, 3, 480, 640)  float32
    Output: (B, 2)            float32
    """
    def __init__(self):
        super().__init__()

        self.backbone = nn.Sequential(
            # 3 x 480 x 640
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # -> 32 x 240 x 320
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64 x 120 x 160
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # collapse spatial dims
            nn.AdaptiveAvgPool2d((1, 1)),  # -> 64 x 1 x 1
        )

        self.head = nn.Sequential(
            nn.Flatten(),                 # -> 64
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 2),            # -> 2 floats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x