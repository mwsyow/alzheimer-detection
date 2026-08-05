import torch
from torch import nn


class Simple3DCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        channels: list[int] = None,
        kernel_size: int | tuple[int, int, int] = 3,
        padding: int | tuple[int, int, int] = 1,
        pool_kernel_size: int | tuple[int, int, int] = 2,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        channels = channels or [16, 32, 64]

        blocks = []
        current_channels = in_channels
        for out_channels in channels:
            blocks.append(
                nn.Conv3d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            if use_batch_norm:
                blocks.append(nn.BatchNorm3d(out_channels))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool3d(pool_kernel_size))
            if dropout > 0:
                blocks.append(nn.Dropout3d(dropout))
            current_channels = out_channels

        blocks.append(nn.AdaptiveAvgPool3d(1))
        self.net = nn.Sequential(*blocks)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(config):
    model_config = config["model"]
    if model_config["name"] != "Simple3DCNN":
        raise ValueError(f"Unsupported model: {model_config['name']}")
    return Simple3DCNN(**model_config.get("params", {}))
