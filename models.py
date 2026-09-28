"""Channel estimation networks: SRCNN and DnCNN (as used in ChannelNet)."""

import torch.nn as nn


class SRCNN(nn.Module):
    """Super-resolution CNN that refines the interpolated channel estimate."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)


class DNCNN(nn.Module):
    """Denoising CNN with residual learning: predicts the noise and subtracts it."""

    def __init__(self, depth=20, n_channels=64):
        super().__init__()
        layers = [nn.Conv2d(1, n_channels, kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(n_channels, eps=1e-3),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(n_channels, 1, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return x - self.dncnn(x)


MODEL_TYPES = ('SRCNN', 'DNCNN')


def build_model(model_type):
    """Instantiate a channel estimator by name."""
    if model_type == 'SRCNN':
        return SRCNN()
    if model_type == 'DNCNN':
        return DNCNN(depth=20, n_channels=64)
    raise ValueError(f"Unknown model type: {model_type}. Choose one of {MODEL_TYPES}.")
