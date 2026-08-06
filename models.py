import torch
from monai.networks.nets import DenseNet121 as BaseDenseNet121
from torch import nn


class PretrainedMixin(nn.Module):
    def load_pretrained_weights(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location="cpu")
        model_state_dict = self.state_dict()
        compatible_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state_dict and value.shape == model_state_dict[key].shape
        }
        self.load_state_dict(compatible_state_dict, strict=False)

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.class_layers.out.parameters():
            param.requires_grad = True


class DenseNet121(BaseDenseNet121, PretrainedMixin):
    pass


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
        if len(channels) == 1 and isinstance(channels[0], list):
            channels = channels[0]
        if not all(isinstance(channel, int) for channel in channels):
            raise TypeError(f"channels must be a list of ints, got {channels!r}")
        if not isinstance(dropout, int | float):
            raise TypeError(f"dropout must be a number, got {dropout!r}")

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
    model_name = model_config["name"]
    params = dict(model_config.get("params", {}))

    if model_name == "Simple3DCNN":
        model = Simple3DCNN(**params)
    elif model_name == "DenseNet121":
        allowed_params = {
            "spatial_dims",
            "in_channels",
            "out_channels",
            "num_classes",
            "init_features",
            "growth_rate",
            "block_config",
            "bn_size",
            "act",
            "norm",
            "dropout_prob",
        }
        params = {key: value for key, value in params.items() if key in allowed_params}
        spatial_dims = params.pop("spatial_dims", 3)
        in_channels = params.pop("in_channels", 1)
        out_channels = params.pop("out_channels", params.pop("num_classes", 2))
        model = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **params,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pretrained = dict(model_config.get("pretrained", {}))
    pretrained_enabled = pretrained.get("enabled", pretrained.get("enable", False))
    if pretrained_enabled and PretrainedMixin in model.__class__.mro():
        pretrained_weights_path = pretrained.get("pretrained_weights_path")
        if pretrained_weights_path:
            model.load_pretrained_weights(pretrained_weights_path)
            freeze_backbone = pretrained.get("freeze_backbone", True)
            if freeze_backbone:
                model.freeze_backbone()
    return model
