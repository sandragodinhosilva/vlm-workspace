"""
Tests for cleanup_checkpoints.py hardening:

BUG 1 — safe run-key resolution (no longest-prefix mis-match). The real-world
collision is `..._merged_1805` (step_2558) vs the longer, DIFFERENT
`..._merged_1805_binary_aux12k_union` (step_4127); the union run must resolve to
its OWN keeper, never to the shorter run's step.

BUG 2 — export-verified guard: never delete a run's checkpoints unless its keeper
step is verified present as an HF export under the models dir.

Run:
  /home/sgsilva/vlm-post-training-home-venv/bin/python -m pytest \
      /home/sgsilva/utilities/cleanup/test_cleanup_checkpoints.py -v
"""

import json
from pathlib import Path

import pytest

import cleanup_checkpoints as cc


# --- synthetic board (mirrors the real collision case) -----------------------

UNION_PREFIX = "qwen35-27b-oracle-obs-merged-1805-binary-aux12k-union-sft"
BARE_PREFIX = "qwen35-27b-oracle-obs-merged-1805"

# (run dir name on disk, step subdirs to create)
DISK_RUNS = {
    "sft_qwen35_27b_oracle_obs_merged_1805": ["step_2000", "step_2558"],
    "sft_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union": [
        "step_3500",
        "step_4000",
        "step_4127",
    ],
}


@pytest.fixture
def synthetic_board(tmp_path, monkeypatch):
    """Build a synthetic master board (CSV + master_models.json) and point the
    module's source globals at it. Returns the tmp model dir path so individual
    tests can choose whether the keeper export is present."""
    board_dir = tmp_path / "master"
    board_dir.mkdir()

    # CSV: Model Path column is index 1 (col 0 is the display name).
    csv_path = board_dir / "eval_master_27b.csv"
    rows = [
        "Display,Model Path,Score",
        f"bare,/mnt/data/sgsilva/models/{BARE_PREFIX}-step2558,0.70",
        f"union,/mnt/data/sgsilva/models/{UNION_PREFIX}-step4127,0.73",
    ]
    csv_path.write_text("\n".join(rows) + "\n")

    json_path = tmp_path / "master_models.json"
    json_path.write_text(
        json.dumps(
            {
                "models": [
                    {"pattern": f"{BARE_PREFIX}-step2558"},
                    {"pattern": f"{UNION_PREFIX}-step4127"},
                ]
            }
        )
    )

    monkeypatch.setattr(cc, "MASTER_CSV_DIR", board_dir)
    monkeypatch.setattr(cc, "MASTER_CSV_FILES", ("eval_master_27b.csv",))
    monkeypatch.setattr(cc, "MASTER_MODELS_JSON_CANDIDATES", (json_path,))

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


def _build_results_dir(tmp_path) -> Path:
    results = tmp_path / "checkpoints"
    results.mkdir()
    for run, steps in DISK_RUNS.items():
        for step in steps:
            d = results / run / step
            d.mkdir(parents=True)
            (d / "model.safetensors").write_bytes(b"x" * 1024)
    return results


# --- BUG 1: resolution ---------------------------------------------------------


def test_union_resolves_to_own_step_not_bare(synthetic_board):
    run_best, _ = cc.detect_best_checkpoints_from_master_board()

    union_dir = "sft_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union"
    union_key = cc._resolve_run_key(union_dir, run_best)
    assert union_key is not None, "union run must resolve to a board key"
    assert run_best[union_key] == "step_4127", (
        "(a) union run must keep step_4127, NOT the shorter _1805 run's step_2558; "
        f"got {run_best.get(union_key)}"
    )


def test_bare_1805_resolves_to_2558(synthetic_board):
    run_best, _ = cc.detect_best_checkpoints_from_master_board()

    bare_dir = "sft_qwen35_27b_oracle_obs_merged_1805"
    bare_key = cc._resolve_run_key(bare_dir, run_best)
    assert bare_key is not None
    assert run_best[bare_key] == "step_2558", (
        "(b) bare _1805 run must keep step_2558; got {}".format(run_best.get(bare_key))
    )


# --- BUG 2: export-verified guard ---------------------------------------------


def test_union_keeps_all_when_export_missing(tmp_path, synthetic_board):
    """(c) With NO exported model present, the union run must keep ALL its
    checkpoints (export guard trips) and delete nothing."""
    models_dir = synthetic_board  # empty
    results = _build_results_dir(tmp_path)
    run_best, _ = cc.detect_best_checkpoints_from_master_board()

    stats = cc.cleanup_checkpoints(
        results,
        run_best,
        dry_run=True,
        models_dir=models_dir,
        skip_export_check=False,
    )

    union_steps = {
        str(results / "sft_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union" / s)
        for s in DISK_RUNS["sft_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union"]
    }
    # No union step was deleted, and all three were kept.
    assert union_steps.isdisjoint(set(stats["deleted_dirs"])), "guard must not delete union steps"
    assert union_steps.issubset(set(stats["kept_dirs"])), "guard must keep all union steps"


def test_union_prunes_siblings_when_export_present(tmp_path, synthetic_board):
    """(d) With the keeper export present, the union run marks step_3500 +
    step_4000 for deletion and keeps step_4127."""
    models_dir = synthetic_board
    # Materialize the keeper export only.
    (models_dir / f"{UNION_PREFIX}-step4127").mkdir()

    results = _build_results_dir(tmp_path)
    run_best, _ = cc.detect_best_checkpoints_from_master_board()

    stats = cc.cleanup_checkpoints(
        results,
        run_best,
        dry_run=True,
        models_dir=models_dir,
        skip_export_check=False,
    )

    base = results / "sft_qwen35_27b_oracle_obs_merged_1805_binary_aux12k_union"
    # In dry-run, deletions are NOT recorded in deleted_dirs (only space_freed),
    # so assert via kept_dirs: keeper kept, siblings NOT kept.
    kept = set(stats["kept_dirs"])
    assert str(base / "step_4127") in kept, "keeper step_4127 must be kept"
    assert str(base / "step_3500") not in kept, "step_3500 must be a deletion candidate"
    assert str(base / "step_4000") not in kept, "step_4000 must be a deletion candidate"
    # And the dry-run accounted for freed space (the two sibling dirs).
    assert stats["space_freed"] > 0, "dry-run should report reclaimable space"
