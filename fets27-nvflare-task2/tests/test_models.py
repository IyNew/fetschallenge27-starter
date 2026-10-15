from __future__ import annotations

import pytest

from fets27_classification.cohort_registry import get_cohort_spec
from fets27_classification.models import (
    GroupNormResNet50Classifier,
    ResNet50Classifier,
    build_model,
    create_model_for_cohort,
)

torch = pytest.importorskip("torch")
pytest.importorskip("monai")


def test_locked_model_is_group_norm_resnet50():
    spec = get_cohort_spec("brain_tumor")
    model = create_model_for_cohort(spec.name)

    assert isinstance(model, GroupNormResNet50Classifier)
    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )
    assert (
        sum(1 for module in model.modules() if isinstance(module, torch.nn.GroupNorm))
        > 1
    )
    assert not any(
        key.endswith("num_batches_tracked") for key in model.state_dict()
    )


def test_batch_norm_variant_keeps_its_running_statistics():
    model = ResNet50Classifier()

    assert any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.modules()
    )
    assert any(key.endswith("num_batches_tracked") for key in model.state_dict())


def test_locked_model_state_dict_round_trips_between_instances():
    first = create_model_for_cohort("brain_tumor")
    second = create_model_for_cohort("brain_tumor")

    second.load_state_dict(first.state_dict(), strict=True)

    assert set(first.state_dict()) == set(second.state_dict())


def test_locked_model_forward_returns_two_logits():
    model = create_model_for_cohort("brain_tumor")
    model.eval()

    with torch.no_grad():
        output = model(torch.zeros(2, 4, 32, 32, 32))

    assert output.shape == (2, 2)


def test_cohort_spec_matches_model_inputs():
    spec = get_cohort_spec("brain_tumor")
    model = create_model_for_cohort(spec.name)
    first_conv = next(
        module for module in model.modules() if isinstance(module, torch.nn.Conv3d)
    )

    assert spec.in_channels == first_conv.in_channels
    assert spec.num_classes == 2
    assert spec.spatial_size == (96, 96, 96)


def test_build_model_supports_centralized_variants():
    assert isinstance(
        build_model({"model": {"name": "resnet50_3d"}}), ResNet50Classifier
    )
    assert isinstance(
        build_model({"model": {"name": "resnet50_groupnorm_3d"}}),
        GroupNormResNet50Classifier,
    )


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model({"model": {"name": "not_a_model"}})


def test_unknown_cohort_raises():
    with pytest.raises(KeyError, match="Unknown cohort"):
        create_model_for_cohort("meningioma_only")
