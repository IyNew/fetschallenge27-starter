from __future__ import annotations

import csv

from conftest import make_test_dir
from fets27_challenge.evaluation import (
    CohortScore,
    SiteScore,
    resolve_best_model_path,
    should_promote_final,
    write_summary,
)


def test_resolve_best_model_path_prefers_best_filename():
    job_root = make_test_dir("evaluation-path") / "job"
    best_path = (
        job_root / "server" / "simulate_job" / "app_server" / "best_FL_global_model.pt"
    )
    fallback_path = (
        job_root / "server" / "simulate_job" / "app_server" / "FL_global_model.pt"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_bytes(b"best")
    fallback_path.write_bytes(b"fallback")

    resolved = resolve_best_model_path(job_root)
    assert resolved == best_path


def test_should_promote_final_only_when_final_is_better():
    assert should_promote_final(0.70, 0.71)
    assert not should_promote_final(0.71, 0.70)
    assert not should_promote_final(0.70, 0.70)


def _cohort_score(**overrides) -> CohortScore:
    defaults = {
        "cohort": "brain_tumor",
        "display_name": "FeTS 2027 Brain Tumor",
        "cohort_score": 0.7,
        "best_model_path": "best_FL_global_model.pt",
        "job_workspace": "job",
        "site_scores": [SiteScore(site_name="site-1", val_dice=0.7)],
    }
    defaults.update(overrides)
    return CohortScore(**defaults)


def test_cohort_score_records_final_round_fields():
    score = _cohort_score()
    assert score.final_round_score is None
    assert score.best_model_source == "selector"

    promoted = _cohort_score(final_round_score=0.8, best_model_source="final_round")
    payload = promoted.to_dict()
    assert payload["final_round_score"] == 0.8
    assert payload["best_model_source"] == "final_round"


def test_write_summary_includes_final_round_columns():
    output_dir = make_test_dir("evaluation-summary")
    score = _cohort_score(final_round_score=0.8, best_model_source="final_round")

    json_path, csv_path = write_summary(output_dir, mode="local", cohort_scores=[score])

    import json

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    cohort = payload["cohorts"][0]
    assert cohort["final_round_score"] == 0.8
    assert cohort["best_model_source"] == "final_round"

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["final_round_score"] == "0.8"
    assert rows[0]["best_model_source"] == "final_round"
