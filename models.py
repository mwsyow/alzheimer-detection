import torch
from monai.networks.nets import DenseNet121 as BaseDenseNet121
from monai.networks.nets import EfficientNetBN as BaseEfficientNetBN
from monai.networks.nets import ResNet as BaseResNet
from monai.networks.nets.resnet import ResNetBlock, get_inplanes
from torch import nn


class PretrainedMixin(nn.Module):
    # Dotted path to the final linear layer, the only module freeze_backbone leaves
    # trainable. Each architecture names it differently.
    classifier_path: str = ""

    def load_pretrained_weights(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location="cpu")
        model_state_dict = self.state_dict()
        compatible_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state_dict and value.shape == model_state_dict[key].shape
        }
        self.load_state_dict(compatible_state_dict, strict=False)

    def classifier(self) -> nn.Module:
        module: nn.Module = self
        for attribute in self.classifier_path.split("."):
            module = getattr(module, attribute)
        return module

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classifier().parameters():
            param.requires_grad = True


class DenseNet121(BaseDenseNet121, PretrainedMixin):
    classifier_path = "class_layers.out"


class ResNet10(BaseResNet, PretrainedMixin):
    """MONAI ResNet-10 — the shallowest 3D ResNet, 14.4M parameters against
    DenseNet121's 11.2M. ResNet-18 is the next one up at 33.2M."""

    classifier_path = "fc"

    def __init__(
        self,
        spatial_dims: int = 3,
        n_input_channels: int = 1,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(
            block=ResNetBlock,
            layers=[1, 1, 1, 1],
            block_inplanes=get_inplanes(),
            spatial_dims=spatial_dims,
            n_input_channels=n_input_channels,
            num_classes=num_classes,
            **kwargs,
        )


class EfficientNetB0(BaseEfficientNetBN, PretrainedMixin):
    """MONAI EfficientNet-B3 at 12.1M parameters, the variant closest to
    DenseNet121's 11.2M (B2 is 8.7M, B0 4.7M). Depth and resolution scaling make it
    slower per step than the parameter count suggests."""

    classifier_path = "_fc"

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 2,
        model_name: str = "efficientnet-b0",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            # ImageNet weights exist for 2D only, and are not what pretrained.enabled
            # in the config means.
            pretrained=False,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs,
        )


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

    # Params are passed straight to the constructor -- no allow-list. A key the
    # model does not accept is a TypeError at build time rather than a silent
    # no-op that costs a full run to discover.
    if model_name == "Simple3DCNN":
        model = Simple3DCNN(**params)
    elif model_name == "DenseNet121":
        spatial_dims = params.pop("spatial_dims", 3)
        in_channels = params.pop("in_channels", 1)
        out_channels = params.pop("out_channels", params.pop("num_classes", 2))
        model = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            **params,
        )
    elif model_name == "ResNet10":
        spatial_dims = params.pop("spatial_dims", 3)
        # in_channels for parity with the other models; MONAI's ResNet spells it
        # n_input_channels.
        n_input_channels = params.pop("n_input_channels", params.pop("in_channels", 1))
        num_classes = params.pop("num_classes", params.pop("out_channels", 2))
        model = ResNet10(
            spatial_dims=spatial_dims,
            n_input_channels=n_input_channels,
            num_classes=num_classes,
            **params,
        )
    elif model_name == "EfficientNetB0":
        spatial_dims = params.pop("spatial_dims", 3)
        in_channels = params.pop("in_channels", 1)
        num_classes = params.pop("num_classes", params.pop("out_channels", 2))
        model = EfficientNetB0(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
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
