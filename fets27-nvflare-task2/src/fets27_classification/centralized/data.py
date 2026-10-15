"""Centralized data loading built on the locked runtime datalists.

The centralized trainer reads the same ``<data-root>/<cohort>/datalist`` JSON
files produced by ``scripts/prepare_partition_datalists.py`` as the NVFLARE
clients, so there is exactly one data contract for the whole project.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from ..data_pipeline import (
    CacheDataset,
    DataLoader,
    Dataset,
    build_classification_transforms,
    compute_class_weights,
    load_datalist_cases,
    require_runtime_dependencies,
)

DEFAULT_COHORT = "brain_tumor"
DEFAULT_SITE = "site-All"


def resolve_datalist_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """Return the datalist JSON and dataset base directory for a config.

    Args:
        config: Loaded configuration mapping with a ``data`` section.

    Returns:
        A ``(datalist_json_path, dataset_base_dir)`` tuple.
    """
    data = config["data"]
    root = Path(data["data_root"])
    cohort = data.get("cohort", DEFAULT_COHORT)
    site = data.get("site", DEFAULT_SITE)
    cohort_root = root / cohort
    return cohort_root / "datalist" / f"{site}.json", cohort_root / "dataset"


def load_cases(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Load the centralized training and validation cases from a datalist.

    Args:
        config: Loaded configuration mapping with a ``data`` section.

    Returns:
        A mapping with ``train`` and ``validation`` case lists.

    Raises:
        FileNotFoundError: If the configured datalist does not exist.
        ValueError: If either split is empty.
    """
    require_runtime_dependencies()
    datalist_path, dataset_base_dir = resolve_datalist_paths(config)
    if not datalist_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared datalist {datalist_path}. Run "
            "scripts/prepare_partition_datalists.py or prepare-assets first."
        )
    cases = {
        split: load_datalist_cases(
            datalist_json_path=str(datalist_path),
            dataset_base_dir=str(dataset_base_dir),
            data_list_key=split,
        )
        for split in ("training", "validation")
    }
    if not cases["training"]:
        raise ValueError(f"No training cases found in {datalist_path}")
    if not cases["validation"]:
        raise ValueError(f"No validation cases found in {datalist_path}")
    return {"train": cases["training"], "validation": cases["validation"]}


def audit_cases(cases: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Return serializable partition and label counts for a loaded case set.

    Args:
        cases: Mapping produced by :func:`load_cases`.

    Returns:
        A dictionary with per-split case counts and label histograms.
    """
    return {
        split: {
            "count": len(items),
            "labels": dict(sorted(Counter(int(x["label"]) for x in items).items())),
        }
        for split, items in cases.items()
    }


def build_loaders(cases: dict[str, list[dict[str, Any]]], config: dict[str, Any]):
    """Build centralized train/validation loaders and class weights.

    Args:
        cases: Mapping produced by :func:`load_cases`.
        config: Loaded configuration mapping.

    Returns:
        A ``(train_loader, valid_loader, class_weights)`` tuple.
    """
    require_runtime_dependencies()
    data_cfg, train_cfg = config["data"], config["training"]
    spatial_size = tuple(data_cfg["spatial_size"])
    train_transform = build_classification_transforms(spatial_size, train=True)
    valid_transform = build_classification_transforms(spatial_size, train=False)
    cache_rate = float(data_cfg.get("cache_rate", 0.0))
    cache_workers = int(data_cfg.get("cache_workers", 1))
    if cache_rate > 0:
        train_set = CacheDataset(
            cases["train"],
            transform=train_transform,
            cache_rate=cache_rate,
            num_workers=cache_workers,
        )
        valid_set = CacheDataset(
            cases["validation"],
            transform=valid_transform,
            cache_rate=cache_rate,
            num_workers=cache_workers,
        )
    else:
        train_set = Dataset(cases["train"], transform=train_transform)
        valid_set = Dataset(cases["validation"], transform=valid_transform)

    workers = int(data_cfg.get("num_workers", 4))
    common: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(_torch().cuda.is_available()),
    }
    if workers:
        common.update({"persistent_workers": True, "prefetch_factor": 2})
    train_loader = DataLoader(
        train_set, batch_size=int(train_cfg["batch_size"]), shuffle=True, **common
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
        shuffle=False,
        **common,
    )
    class_weights = compute_class_weights(
        [int(case["label"]) for case in cases["train"]]
    )
    return train_loader, valid_loader, class_weights


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    Args:
        seed: The seed applied to every generator.
    """
    import numpy as np

    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def _torch():
    import torch

    return torch
