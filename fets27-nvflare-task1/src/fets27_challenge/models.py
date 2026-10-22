"""Locked model definitions."""

from __future__ import annotations

from typing import Any

try:
    import torch.nn as nn
    from monai.networks.nets import UNet
    from monai.networks.nets.segresnet import SegResNet
except ImportError:  # pragma: no cover - exercised only with full runtime deps
    nn = None
    UNet = None
    SegResNet = None


def _require_monai():
    """Ensure that MONAI is available, raising an ImportError if it is not."""
    if SegResNet is None or UNet is None:
        raise ImportError("MONAI is required to instantiate the challenge models.")


class SerializableSegResNet(SegResNet if SegResNet is not None else object):
    """SegResNet wrapper that preserves constructor arguments for NVFLARE."""

    def __init__(
        self,
        *,
        blocks_down: tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: tuple[int, ...] = (1, 1, 1),
        init_filters: int = 8,
        in_channels: int = 4,
        out_channels: int = 3,
        dropout_prob: float = 0.2,
    ):
        """Initialize the SerializableSegResNet model.

        Preserves configuration attributes as instance attributes for serialization/inspection in NVFLARE.
        """
        _require_monai()
        super().__init__(
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob


class GliomaSegResNet(SerializableSegResNet):
    """Specific SegResNet configuration used for glioma segmentation."""

    def __init__(self):
        """Initialize the GliomaSegResNet model with preset hyperparameters."""
        super().__init__(
            in_channels=4, out_channels=3, init_filters=8, dropout_prob=0.2
        )


class FedUNet(nn.Module if nn is not None else object):
    """MONAI 3D UNet from the FeTS2024-Uncertainty baseline.

    Settings mirror the centralized path-E run
    (``stage1-E-fedunet-96-20ep`` / ``configs/models/fedunet.yaml``): 4 input
    channels, 3 output channels, channels ``(32, 64, 128, 256, 512)``, four
    stride-2 downsamplings, batch norm, dropout 0.3, and three residual units
    per level.

    The published baseline uses 4 output channels and a softmax head. This
    implementation defaults to 3 channels so it matches the current TC/WT/ET
    sigmoid Dice pipeline used by path E.
    """

    def __init__(
        self,
        *,
        in_channels: int = 4,
        out_channels: int = 3,
        norm: str = "batch",
    ):
        """Initialize the path-E FedUNet configuration.

        Args:
            in_channels: Number of input MRI modalities.
            out_channels: Number of output channels (3 for path E; 4 is
                reserved for the published softmax recipe).
            norm: MONAI normalization type; path E uses ``batch``.

        Raises:
            ValueError: If the output channel count is unsupported.
        """
        _require_monai()
        if out_channels not in (3, 4):
            raise ValueError(
                "FedUNet out_channels must be 3 (path E) or 4 "
                "(published softmax recipe)."
            )
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm_name = str(norm)
        self.channels = (32, 64, 128, 256, 512)
        self.strides = (2, 2, 2, 2)
        self.dropout = 0.3
        self.num_res_units = 3
        self.unet = UNet(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            up_kernel_size=3,
            channels=self.channels,
            strides=self.strides,
            norm=self.norm_name,
            dropout=self.dropout,
            num_res_units=self.num_res_units,
        )

    def forward(self, x):
        """Run the UNet forward pass."""
        return self.unet(x)


MODEL_CLASS_BY_COHORT = {
    "glioma": GliomaSegResNet,
}

MODEL_CLASS_BY_NAME = {
    "glioma_segresnet": GliomaSegResNet,
    "fedunet": FedUNet,
}


def create_model_for_cohort(
    cohort_name: str, model_name: str | None = None
) -> Any:
    """Create a new model instance appropriate for the specified cohort.

    Args:
        cohort_name: Name of the cohort (e.g. 'glioma').
        model_name: Optional explicit model name (e.g. 'fedunet'). When
            omitted, the cohort's locked default model is used.

    Returns:
        An instance of the neural network model.

    Raises:
        KeyError: If the cohort or explicit model name is unknown.
    """
    if model_name is not None:
        try:
            model_cls = MODEL_CLASS_BY_NAME[model_name]
        except KeyError as exc:
            raise KeyError(f"Unknown model {model_name!r}.") from exc
        return model_cls()
    try:
        model_cls = MODEL_CLASS_BY_COHORT[cohort_name]
    except KeyError as exc:
        raise KeyError(f"Unknown cohort {cohort_name!r}.") from exc
    return model_cls()
