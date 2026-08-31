import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch
from monai.networks.nets import DenseNet121 as BaseDenseNet121
from monai.networks.nets import EfficientNetBN as BaseEfficientNetBN
from monai.networks.nets.efficientnet import EfficientNet as BaseEfficientNet
from monai.networks.nets.efficientnet import efficientnet_params
from monai.networks.nets import ResNet as BaseResNet
from monai.networks.nets.resnet import ResNetBlock, get_inplanes
from torch import nn

MEDICALNET_RESNET10_REPO = "TencentMedicalNet/MedicalNet-Resnet10"
MEDICALNET_RESNET10_FILENAME = "resnet_10_23dataset.pth"
MEDICALNET_RESNET10_REVISION = "2a0c8cd91b82beb69610b60cb76d9eb8cbf9eac7"
MEDICALNET_RESNET10_SHA256 = (
    "afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_medicalnet_resnet10() -> Path:
    """Download the immutable MedicalNet ResNet10 checkpoint through HF Hub."""
    try:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=MEDICALNET_RESNET10_REPO,
                filename=MEDICALNET_RESNET10_FILENAME,
                revision=MEDICALNET_RESNET10_REVISION,
            )
        )
    except Exception as error:
        raise RuntimeError(
            "Could not download MedicalNet ResNet10 from Hugging Face Hub "
            f"({MEDICALNET_RESNET10_REPO}@{MEDICALNET_RESNET10_REVISION}). "
            "Check compute-node internet access and HF_HOME."
        ) from error

    actual_sha256 = _sha256(path)
    if actual_sha256 != MEDICALNET_RESNET10_SHA256:
        raise RuntimeError(
            f"MedicalNet checkpoint checksum mismatch at {path}: expected "
            f"{MEDICALNET_RESNET10_SHA256}, got {actual_sha256}. Remove the "
            "cached file and retry."
        )
    print(
        "MedicalNet pretrained weights: "
        f"{MEDICALNET_RESNET10_REPO}@{MEDICALNET_RESNET10_REVISION} -> {path} "
        f"(sha256={actual_sha256})"
    )
    return path


class PretrainedMixin(nn.Module):
    # Dotted path to the final linear layer, the only module freeze_backbone leaves
    # trainable. Each architecture names it differently.
    classifier_path: str = ""

    def load_pretrained_weights(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, Mapping):
            state_dict = state_dict.get(
                "state_dict", state_dict.get("model_state_dict", state_dict)
            )
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Expected a state dict in {weights_path}, got {type(state_dict)}"
            )
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
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
        self._backbone_frozen = True
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classifier().parameters():
            param.requires_grad = True
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_backbone_frozen", False):
            classifier = self.classifier()
            for module in self.modules():
                if module is not self and module is not classifier:
                    module.eval()
            classifier.train()
        return self


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

    def load_medicalnet_weights(self):
        checkpoint_path = download_medicalnet_resnet10()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("state_dict"), Mapping
        ):
            raise RuntimeError(
                f"MedicalNet checkpoint {checkpoint_path} has no state_dict mapping."
            )

        pretrained = {
            key.removeprefix("module."): value
            for key, value in checkpoint["state_dict"].items()
        }
        model_state = self.state_dict()
        classifier_keys = {
            f"{self.classifier_path}.weight",
            f"{self.classifier_path}.bias",
        }
        expected_backbone = set(model_state) - classifier_keys
        checkpoint_keys = set(pretrained)
        missing = sorted(expected_backbone - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - expected_backbone)
        mismatched = sorted(
            key
            for key in expected_backbone & checkpoint_keys
            if pretrained[key].shape != model_state[key].shape
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "MedicalNet ResNet10 is incompatible with the constructed backbone: "
                f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}."
            )

        incompatible = self.load_state_dict(pretrained, strict=False)
        if (
            set(incompatible.missing_keys) != classifier_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "MedicalNet load did not leave exactly the classifier uninitialized: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}."
            )


# The EfficientNet-B block topology from the paper, identical for every B variant --
# only the width/depth/resolution coefficients differ, and those come from MONAI's own
# efficientnet_params table. MONAI hardcodes these strings inside EfficientNetBN's
# __init__ rather than exposing them, so EfficientNet has to repeat them. Verified in
# tests/test_efficientnet_dropout.py: building EfficientNet with these plus the table
# values reproduces EfficientNetBN exactly for b0-b3 -- same state-dict keys, same
# shapes, same per-block drop-connect schedule.
EFFICIENTNET_BLOCKS_ARGS = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]


class EfficientNetBN(BaseEfficientNetBN, PretrainedMixin):
    """MONAI's variant wrapper: pick a B number and go.

    Use this when the B number is the only capacity knob you need. `norm` passes
    through to all 49 normalisation layers, but dropout_rate and drop_connect_rate do
    NOT -- EfficientNetBN reads them from the per-variant table and does not accept
    them, so they stay at the table's values and never appear in the run config. Use
    EfficientNet when those matter.
    """

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


class EfficientNet(BaseEfficientNet, PretrainedMixin):
    """The full-control form: every regulariser and scaling coefficient is an argument.

    Subclasses MONAI's EfficientNet rather than its EfficientNetBN wrapper, so
    dropout_rate and drop_connect_rate are real constructor arguments instead of
    table lookups. Both matter on this task: the network reaches train AUC 0.99+
    against val 0.74, so it overfits, and stochastic depth over the MBConv blocks is a
    strong regulariser at 159 training volumes.

    width_coefficient and depth_coefficient are exposed for the same reason -- a
    continuous capacity axis, finer than the B number, which only moves in the table's
    steps and measured monotonically worse here.

    model_name seeds the defaults from MONAI's table, and every other argument
    defaults to None meaning "use that seed". So EfficientNet(model_name="...") with
    nothing else builds exactly what EfficientNetBN would have built.
    """

    classifier_path = "_fc"

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 2,
        model_name: str = "efficientnet-b0",
        dropout_rate: float | None = None,
        drop_connect_rate: float | None = None,
        width_coefficient: float | None = None,
        depth_coefficient: float | None = None,
        image_size: int | None = None,
        blocks_args_str: list[str] | None = None,
        **kwargs,
    ):
        if model_name not in efficientnet_params:
            raise ValueError(
                f"Unknown model_name {model_name!r}. "
                f"Options: {sorted(efficientnet_params)}"
            )
        width, depth, size, dropout, drop_connect = efficientnet_params[model_name]

        def pick(override, default):
            return default if override is None else override

        super().__init__(
            blocks_args_str=blocks_args_str or EFFICIENTNET_BLOCKS_ARGS,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=pick(width_coefficient, width),
            depth_coefficient=pick(depth_coefficient, depth),
            dropout_rate=pick(dropout_rate, dropout),
            # The table's training resolution, used only for the static padding
            # calculation. EfficientNetBN passes the same value, so this matches it.
            image_size=pick(image_size, size),
            drop_connect_rate=pick(drop_connect_rate, drop_connect),
            **kwargs,
        )


class EfficientNetB0(EfficientNetBN):
    """Kept so existing configs and checkpoint metadata keep resolving.

    Every run before 2026-08-24 recorded model.name "EfficientNetB0", and evaluate.py
    rebuilds the architecture from that metadata, so renaming it would make those
    checkpoints unloadable. New configs should say "EfficientNetBN" or "EfficientNet".
    """


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
        self._initialize_weights()

    def _initialize_weights(self):
        """Explicit baseline initialization; MONAI architectures are untouched."""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(config, initialize_pretrained: bool = True):
    model_config = config["model"]
    model_name = model_config["name"]
    params = dict(model_config.get("params", {}))
    pretrained = dict(model_config.get("pretrained", {}))
    pretrained_enabled = pretrained.get("enabled", pretrained.get("enable", False))
    pretrained_source = pretrained.get("source")

    if pretrained_enabled and pretrained_source == "medicalnet":
        if model_name != "ResNet10":
            raise ValueError(
                "pretrained.source='medicalnet' is supported only by ResNet10"
            )
        if pretrained.get("pretrained_weights_path"):
            raise ValueError(
                "MedicalNet is configured as a Hugging Face source; remove "
                "pretrained_weights_path."
            )

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
    elif model_name in ("EfficientNet", "EfficientNetBN", "EfficientNetB0"):
        spatial_dims = params.pop("spatial_dims", 3)
        in_channels = params.pop("in_channels", 1)
        num_classes = params.pop("num_classes", params.pop("out_channels", 2))
        efficientnet_class = {
            "EfficientNet": EfficientNet,
            "EfficientNetBN": EfficientNetBN,
            "EfficientNetB0": EfficientNetB0,
        }[model_name]
        model = efficientnet_class(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            **params,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if (
        initialize_pretrained
        and pretrained_enabled
        and isinstance(model, PretrainedMixin)
    ):
        pretrained_weights_path = pretrained.get("pretrained_weights_path")
        if pretrained_source == "medicalnet":
            model.load_medicalnet_weights()
        elif pretrained_weights_path:
            model.load_pretrained_weights(pretrained_weights_path)
        else:
            raise ValueError(
                "Pretraining is enabled but neither a supported source nor "
                "pretrained_weights_path was configured."
            )

    # Resume/evaluation skip the initial download because a trained state dict is
    # restored immediately afterwards, but a resumed optimizer still needs the same
    # trainable parameter set as the original run.
    if (
        pretrained_enabled
        and isinstance(model, PretrainedMixin)
        and pretrained.get("freeze_backbone", True)
    ):
        model.freeze_backbone()
    return model
