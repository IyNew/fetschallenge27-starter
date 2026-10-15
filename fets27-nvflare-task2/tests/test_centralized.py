from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from conftest import make_test_dir

from fets27_classification.centralized.config import load_config, with_overrides
from fets27_classification.centralized.data import audit_cases, build_loaders, load_cases
from fets27_classification.centralized.sweep import run_sweep
from fets27_classification.centralized.train import run_training
from fets27_classification.synthetic_data import generate_toy_dataset

pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("nibabel")


def _toy_data_root() -> Path:
    root = make_test_dir("centralized-data") / "toy_data"
    generate_toy_dataset(root)
    return root


def _config(data_root: Path, output_dir: Path, **training_overrides) -> dict:
    config = {
        "output_dir": str(output_dir),
        "data": {
            "data_root": str(data_root),
            "cohort": "brain_tumor",
            "site": "site-All",
            "spatial_size": [16, 16, 16],
            "cache_rate": 0.0,
            "cache_workers": 1,
            "num_workers": 0,
        },
        "model": {"name": "resnet50_groupnorm_3d"},
        "training": {
            "seed": 17,
            "device": "cpu",
            "amp": False,
            "epochs": 1,
            "early_stopping_patience": 3,
            "batch_size": 2,
            "eval_batch_size": 2,
            "learning_rate": 0.0001,
            "min_learning_rate": 0.000001,
            "weight_decay": 0.00001,
        },
    }
    config["training"].update(training_overrides)
    return config


def test_load_config_env_override_has_lower_precedence_than_explicit_override(monkeypatch):
    config_path = make_test_dir("centralized-config") / "config.yaml"
    config_path.write_text("data:\n  data_root: /yaml/data\n", encoding="utf-8")
    monkeypatch.setenv("FETS27_DATA_ROOT", "/environment/data")

    assert load_config(config_path)["data"]["data_root"] == "/environment/data"
    assert (
        load_config(config_path, ["data.data_root=/explicit/data"])["data"][
            "data_root"
        ]
        == "/explicit/data"
    )


def test_load_config_rejects_malformed_override():
    config_path = make_test_dir("centralized-config-bad") / "config.yaml"
    config_path.write_text("data:\n  data_root: /yaml/data\n", encoding="utf-8")

    with pytest.raises(ValueError, match="key=value"):
        load_config(config_path, ["nonsense"])


def test_with_overrides_does_not_mutate_the_source_mapping():
    source = {"training": {"epochs": 5}}

    result = with_overrides(source, {"training.epochs": 1})

    assert result["training"]["epochs"] == 1
    assert source["training"]["epochs"] == 5


def test_load_cases_and_audit_cases_use_the_runtime_datalists():
    cases = load_cases(_config(_toy_data_root(), Path("outputs")))

    audit = audit_cases(cases)

    assert audit["train"] == {"count": 4, "labels": {0: 2, 1: 2}}
    assert audit["validation"] == {"count": 2, "labels": {0: 1, 1: 1}}


def test_load_cases_requires_a_prepared_datalist():
    root = make_test_dir("centralized-missing") / "empty"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing prepared datalist"):
        load_cases(_config(root, Path("outputs")))


def test_load_cases_requires_a_validation_split():
    root = _toy_data_root()
    datalist_path = root / "brain_tumor" / "datalist" / "site-All.json"
    payload = json.loads(datalist_path.read_text(encoding="utf-8"))
    validation = payload.pop("validation")
    payload["training"] = [validation[0]]
    datalist_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="No validation cases"):
        load_cases(_config(root, Path("outputs")))


def test_build_loaders_returns_batches_and_class_weights():
    config = _config(_toy_data_root(), Path("outputs"))
    cases = load_cases(config)

    train_loader, valid_loader, class_weights = build_loaders(cases, config)

    batch = next(iter(train_loader))
    assert batch["image"].shape == (2, 4, 16, 16, 16)
    assert len(valid_loader) == 1
    assert class_weights.tolist() == [1.0, 1.0]


def test_run_training_writes_summary_with_mcc():
    root = make_test_dir("centralized-train")
    config = _config(_toy_data_root(), root / "outputs")

    summary = run_training(config, "toy-run")

    output_dir = root / "outputs" / "toy-run"
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "summary.json").is_file()
    assert summary["epochs_completed"] == 1
    assert "mcc" in summary["metrics"]

    epoch_records = [
        json.loads(line)
        for line in (output_dir / "epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(epoch_records) == 1
    assert "mcc" in epoch_records[0]

    # Checkpoints are ~185 MB; do not leave them in the working tree.
    shutil.rmtree(root / "outputs", ignore_errors=True)


def test_run_sweep_ranks_a_single_trial():
    root = make_test_dir("centralized-sweep")
    data_root = root / "toy_data"
    generate_toy_dataset(data_root)
    model_config = _config(data_root, root / "outputs")
    base_config_path = root / "model.yaml"
    base_config_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    sweep_path = root / "sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "name": "toy-sweep",
                "base_config": str(base_config_path),
                "run_output_dir": str(root / "outputs"),
                "output_dir": str(root / "sweep-output"),
                "grid": {"training.learning_rate": [0.0001]},
            }
        ),
        encoding="utf-8",
    )

    results = run_sweep(sweep_path)

    assert len(results) == 1
    assert results[0]["run_name"] == "toy-sweep-01"
    assert "mcc" in results[0]
    ranking = json.loads((root / "sweep-output" / "ranking.json").read_text(encoding="utf-8"))
    mean_metrics = json.loads(
        (root / "sweep-output" / "mean_metrics.json").read_text(encoding="utf-8")
    )
    assert ranking[0]["run_name"] == "toy-sweep-01"
    assert "mcc" in mean_metrics
    assert mean_metrics["trial_count"] == 1

    # Each trial writes a full checkpoint; keep the working tree small.
    shutil.rmtree(root / "outputs", ignore_errors=True)
