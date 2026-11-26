from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fets27_challenge.compat import FLMetaKey, FLModel
from fets27_challenge.participant_loader import load_participant_aggregator
from fets27_challenge.reference_aggregators import (
    ClippedMeanAggregator,
    MedianAggregator,
    WeightedFedAvgAggregator,
)


def _model(values, weight):
    return FLModel(
        params={"w": np.asarray(values, dtype=np.float32)},
        params_type="DIFF",
        meta={FLMetaKey.NUM_STEPS_CURRENT_ROUND: weight},
    )


def test_weighted_aggregator_uses_step_counts():
    aggregator = WeightedFedAvgAggregator()
    aggregator.accept_model(_model([1.0, 3.0], 1))
    aggregator.accept_model(_model([3.0, 9.0], 3))
    result = aggregator.aggregate_model()
    np.testing.assert_allclose(
        result.params["w"], np.array([2.5, 7.5], dtype=np.float32)
    )


def test_median_aggregator_is_coordinatewise():
    aggregator = MedianAggregator()
    aggregator.accept_model(_model([1.0, 50.0], 1))
    aggregator.accept_model(_model([2.0, 5.0], 1))
    aggregator.accept_model(_model([3.0, 6.0], 1))
    result = aggregator.aggregate_model()
    np.testing.assert_allclose(
        result.params["w"], np.array([2.0, 6.0], dtype=np.float32)
    )


def test_clipped_mean_aggregator_preserves_shape():
    aggregator = ClippedMeanAggregator(clip_percentile=50.0)
    aggregator.accept_model(_model([1.0, 100.0], 1))
    aggregator.accept_model(_model([2.0, 2.0], 1))
    aggregator.accept_model(_model([3.0, 3.0], 1))
    result = aggregator.aggregate_model()
    assert result.params["w"].shape == (2,)


def test_participant_aggregator_loads_and_matches_contract():
    aggregator = load_participant_aggregator(Path.cwd())
    aggregator.accept_model(_model([1.0, 1.0], 1))
    aggregator.accept_model(_model([3.0, 3.0], 3))
    result = aggregator.aggregate_model()
    np.testing.assert_allclose(
        result.params["w"], np.array([2.5, 2.5], dtype=np.float32)
    )


def test_reference_aggregators_are_nvflare_model_aggregators():
    model_aggregator_module = pytest.importorskip(
        "nvflare.app_common.aggregators.model_aggregator"
    )

    for aggregator_cls in (
        WeightedFedAvgAggregator,
        MedianAggregator,
        ClippedMeanAggregator,
    ):
        assert isinstance(aggregator_cls(), model_aggregator_module.ModelAggregator)


def test_fedavg_recipe_accepts_reference_aggregators():
    torch = pytest.importorskip("torch")
    fedavg_module = pytest.importorskip("nvflare.app_opt.pt.recipes.fedavg")

    model = torch.nn.Linear(4, 2)
    for aggregator_cls in (
        WeightedFedAvgAggregator,
        MedianAggregator,
        ClippedMeanAggregator,
    ):
        aggregator = aggregator_cls()
        recipe = fedavg_module.FedAvgRecipe(
            name="test_reference_aggregator",
            train_script="participant/client.py",
            model=model,
            aggregator=aggregator,
            num_rounds=1,
            min_clients=1,
        )
        assert recipe._get_model_aggregator() is aggregator

