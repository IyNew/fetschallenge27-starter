"""Locked model definitions and the centralized model factory.

The NVFLARE runtime always uses :func:`create_model_for_cohort`, which returns
the locked cohort model. :func:`build_model` is only used by the organizer-side
centralized trainer/sweep tooling and supports the additional BatchNorm variant
for comparison.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised only with full runtime deps
    import torch.nn as nn
    from monai.networks.nets import resnet50
except ImportError:  # pragma: no cover - imported in tests without heavy deps
    nn = None
    resnet50 = None


def _require_model_dependencies():
    """Ensure PyTorch and MONAI are available before building a classifier."""
    if nn is None or resnet50 is None:
        raise ImportError(
            "PyTorch and MONAI are required to instantiate the challenge models."
        )


def _build_resnet50(*, group_norm: bool):
    """Build the MONAI 3D ResNet-50 backbone used by every classifier.

    MONAI's ``resnet50`` factory only forwards its ``norm`` argument to the stem
    normalization; every bottleneck block is hardcoded to BatchNorm. The
    federation-safe variant therefore replaces *all* BatchNorm modules after
    construction so no running statistics or ``num_batches_tracked`` buffers are
    exchanged through NVFLARE's weight-difference transfer.
    """
    kwargs: dict[str, Any] = {
        "spatial_dims": 3,
        "n_input_channels": 4,
        "num_classes": 2,
        "conv1_t_size": 7,
        "conv1_t_stride": 1,
        "no_max_pool": False,
        "shortcut_type": "B",
        "bias_downsample": False,
    }
    network = resnet50(**kwargs)
    if group_norm:
        _replace_batch_norm_with_group_norm(network)
    return network


def _replace_batch_norm_with_group_norm(module, num_groups: int = 16):
    """Recursively replace BatchNorm layers with channel-compatible GroupNorm.

    Args:
        module: Module tree to rewrite in place.
        num_groups: Preferred number of groups; falls back to one group when the
            layer's channel count is not divisible by it.

    Returns:
        The rewritten module tree.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            groups = num_groups if child.num_features % num_groups == 0 else 1
            setattr(
                module,
                name,
                nn.GroupNorm(
                    num_groups=groups,
                    num_channels=child.num_features,
                    eps=child.eps,
                    affine=child.affine,
                ),
            )
        else:
            _replace_batch_norm_with_group_norm(child, num_groups)
    return module


class _ResNet50ClassifierBase(nn.Module if nn is not None else object):
    """Serializable four-channel 3D ResNet-50 binary classifier.

    NVFLARE's model persistor reconstructs the server model from its class and
    constructor arguments, so the locked model exposes a no-required-argument
    constructor and keeps the MONAI factory output as a submodule.
    """

    _group_norm = False

    def __init__(self):
        """Initialize the architecture used by both centralized and FL runs."""
        _require_model_dependencies()
        super().__init__()
        self.network = _build_resnet50(group_norm=self._group_norm)

    def forward(self, inputs):
        """Return binary logits for a batch of 3D four-modality images."""
        return self.network(inputs)


class ResNet50Classifier(_ResNet50ClassifierBase):
    """BatchNorm 3D ResNet-50 kept for the centralized comparison baseline."""

    _group_norm = False


class GroupNormResNet50Classifier(_ResNet50ClassifierBase):
    """GroupNorm 3D ResNet-50 used by the locked federated baseline."""

    _group_norm = True


MODEL_CLASS_BY_COHORT = {
    "brain_tumor": GroupNormResNet50Classifier,
}


def create_model_for_cohort(cohort_name: str):
    """Create the locked model instance for a cohort.

    Args:
        cohort_name: Name of the cohort (e.g. 'brain_tumor').

    Returns:
        An instance of the cohort's locked neural network.

    Raises:
        KeyError: If the cohort name is unknown or has no model registered.
    """
    try:
        model_cls = MODEL_CLASS_BY_COHORT[cohort_name]
    except KeyError as exc:
        raise KeyError(f"Unknown cohort {cohort_name!r}.") from exc
    return model_cls()


def build_model(config: dict[str, Any]):
    """Instantiate the classifier named by a centralized YAML configuration.

    Args:
        config: Loaded YAML mapping with a ``model.name`` entry.

    Returns:
        The configured classifier module.

    Raises:
        ValueError: If ``model.name`` is not supported.
    """
    model = config["model"]
    if model["name"] == "resnet50_3d":
        return ResNet50Classifier()
    if model["name"] == "resnet50_groupnorm_3d":
        return GroupNormResNet50Classifier()
    raise ValueError(f"Unsupported model {model['name']!r}.")
