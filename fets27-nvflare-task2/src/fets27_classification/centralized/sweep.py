"""Sequential YAML-defined hyperparameter sweep runner."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

MEAN_METRIC_KEYS = (
    "balanced_accuracy",
    "accuracy",
    "auroc",
    "macro_f1",
    "sensitivity",
    "specificity",
    "mcc",
)


def run_sweep(sweep_path: str | Path) -> list[dict]:
    """Execute every Cartesian-product trial in a sweep YAML sequentially.

    Args:
        sweep_path: Path to the sweep specification YAML.

    Returns:
        The ranked list of per-trial result rows.

    Raises:
        RuntimeError: If the sweep specification is incomplete or a trial
            produced no summary.
    """
    with Path(sweep_path).open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    for key in ("name", "base_config", "run_output_dir", "output_dir", "grid"):
        if key not in spec:
            raise RuntimeError(f"Sweep specification is missing {key!r}.")
    output = Path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    keys, values = list(spec["grid"]), list(spec["grid"].values())
    results = []
    for index, combination in enumerate(itertools.product(*values), start=1):
        overrides = dict(zip(keys, combination))
        run_name = f"{spec['name']}-{index:02d}"
        command = [
            sys.executable,
            "-m",
            "fets27_classification.cli",
            "train",
            "--config",
            spec["base_config"],
            "--run-name",
            run_name,
        ]
        for key, value in overrides.items():
            command += ["--override", f"{key}={json.dumps(value)}"]
        subprocess.run(command, check=True, env=_subprocess_env())
        summary_path = Path(spec["run_output_dir"]) / run_name / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"No summary was written for trial {run_name}.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append({"run_name": run_name, **overrides, **summary["metrics"]})
        _write_results(output / "results.csv", results)
    results.sort(key=lambda item: item["balanced_accuracy"], reverse=True)
    (output / "ranking.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    aggregate = {
        key: sum(float(row[key]) for row in results) / len(results)
        for key in MEAN_METRIC_KEYS
    }
    aggregate["trial_count"] = len(results)
    (output / "mean_metrics.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    return results


def _subprocess_env() -> dict:
    """Return an environment that can import the package from a dev checkout."""
    env = dict(os.environ)
    src_root = Path(__file__).resolve().parents[2]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(src_root) if not existing else f"{src_root}{os.pathsep}{existing}"
    )
    return env


def _write_results(path: Path, rows: list[dict]) -> None:
    """Write incremental sweep results after each completed trial."""
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the sweep runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args()
    run_sweep(args.sweep)


if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    main()
