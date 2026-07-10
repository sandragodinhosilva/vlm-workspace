#!/usr/bin/env python3
"""Compile eval results from all three pipelines into ONE master CSV — additive, read-only.

Unified results tree (reorg 2026-06-17): /mnt/data/sgsilva/results/{aux,benchmarks,visual_obs,master}.

Sources (NEVER modified):
  1. aux        — PRIMARY: /mnt/data/sgsilva/results/aux/eval_matrix_qwen3.5-4b.csv (+ combined),
                  the rich aux master (train_reasoning, train_sample_count, acc_weighted_3mod,
                  per-task breakdowns). FALLBACK: the per-run multimodal_*.json under
                  aux/evals/.../results/ for runs not yet exported into eval_matrix.
  2. benchmarks — /mnt/data/sgsilva/results/benchmarks/summary[_judge].csv
                  (cols: Model, Reasoning, MMMU-val, Video-MME, VSI-Bench, Test set Acc)
  3. visualobs  — /mnt/data/sgsilva/results/visual_obs/runs/*singlestage*.json | stage2_*.json
                  (metrics.{error_detection_f1, sample_error_detection_f1, overall_severity_accuracy})

Output (additive — originals stay the source of truth):
  /mnt/data/sgsilva/results/master/eval_master.csv          one row per (model, thinking)
  /mnt/data/sgsilva/results/master/eval_master_{4b,9b,27b}.csv  per-base-model, baselines pinned top
  /mnt/data/sgsilva/results/master/runs/<run>/...           copies of each run's aggregate JSON + SUMMARY

This script does NOT run evals, re-score, or touch the per-stage collectors/CSVs. It only reads
their outputs and unifies them. Re-run anytime; it rebuilds the master CSV from scratch.

Usage:
  /home/sgsilva/vlm-post-training-home-venv/bin/python compile_eval_results.py [--no-copy]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
from pathlib import Path

# Reorg 2026-06-17: unified results tree under /mnt/data/sgsilva/results/{aux,benchmarks,visual_obs,master}.
AUX_EVALS = Path("/home/sgsilva/vlm-post-training/aux_tasks/evals")  # symlink -> /results/aux/evals (per-run JSON tree; fallback source)
AUX_MATRIX = Path("/mnt/data/sgsilva/results/aux/eval_matrix.csv")             # rich aux master (combined base models)
AUX_MATRIX_4B = Path("/mnt/data/sgsilva/results/aux/eval_matrix_qwen3.5-4b.csv")  # per-base-model (primary aux source)
BENCH_RESULTS = Path("/mnt/data/sgsilva/results/benchmarks")  # moved here 2026-06-17 (old benchmarks/results back-compat-symlinked)
BENCH_SUMMARY = BENCH_RESULTS / "summary.csv"
BENCH_SUMMARY_JUDGE = BENCH_RESULTS / "summary_judge.csv"
VO_RUNS = Path("/mnt/data/sgsilva/results/visual_obs/runs")    # moved 2026-06-17 (old visual_obs_runs/ back-compat-symlinked)
MASTER_DIR = Path("/mnt/data/sgsilva/results/master")          # renamed from eval_master/ (reorg 2026-06-17)
MASTER_CSV = MASTER_DIR / "eval_master.csv"
COPY_DIR = MASTER_DIR / "runs"

# New-era curation: the LIVE eval_master* files hold a CURATED allowlist of models (the new-era
# board). The full historical dump is frozen under results/master/v1/. The allowlist is
# master_models.json — a list of {pattern, display?, train_reasoning?, note?}; a row is kept iff
# its served path/display matches any pattern's substring (first match supplies the curated
# display + train_reasoning). Edit that JSON to add a model; re-run the compiler.
# If the allowlist file is ABSENT, the compiler falls back to writing ALL rows (legacy behavior).
MODEL_ALLOWLIST = Path("/home/sgsilva/utilities/eval/master_models.json")
ERA_FAMILIES = ("4b", "27b")  # only these families get split files in the new era (2026-06-18)

# VO-usage grounding scorer (2026-07-07): reuse the canonical implementation instead of
# re-implementing the block/answer matching here. Soft dependency — if the repo script is
# missing/broken, the board still builds and the vo_s2_usage_* columns stay blank (loudly).
_VO_USAGE_SCRIPT = Path("/home/sgsilva/vlm-post-training/visual_obs/measure_stage2_vo_usage.py")
try:
    import importlib.util as _ilu
    _vou_spec = _ilu.spec_from_file_location("measure_stage2_vo_usage", _VO_USAGE_SCRIPT)
    _vou = _ilu.module_from_spec(_vou_spec)
    _vou_spec.loader.exec_module(_vou)
except Exception as _vou_err:  # noqa: BLE001 — any import failure just disables the columns
    print(f"[vo-usage] scorer unavailable ({_vou_err}) — vo_s2_usage_* columns will stay BLANK")
    _vou = None

# one row per (model_key, thinking); model_key = the served model basename / display name.
# Column order = logical reading order: IDENTITY -> WHEN -> HEADLINE SCORES (benchmarks +
# aux + visual-obs) -> AUX DETAIL -> TRAINING PROVENANCE -> SOURCE PROVENANCE (bookkeeping).
# ---------------------------------------------------------------------------------------------------
# FULL VO METRIC DETAIL BLOCK — mirrors visual_obs_sft_results_1105_formatted.csv's per-band metric
# layout (cols 9-46), appended AFTER the provenance columns (defined BEFORE FIELDS because FIELDS
# expands it). Emitted for BOTH single-stage (s1) and two-stage (s2); s2 is mostly blank until the
# reasoner sweep fills it. Columns intentionally REPEAT the headline ones (Error-F1/Sample-F1/
# Severity-Acc) so each block is a self-contained mirror of the formatted CSV's band.
# Each entry: (suffix, header, source) where source is a metrics.* key OR a ("raw"/"per_sev", ...) tuple.
# Layout MIRRORS visual_obs_sft_results_1105_formatted.csv EXACTLY, including the BLANK separator
# columns between sub-blocks and the (unpopulated) "Variability / Avg Dist Exercise" column (which
# the result JSON doesn't carry — kept blank to preserve the disposition). source: a metrics.* key
# (×100 pct), a ("raw", key) tuple (no ×100: MAE/Pearson/Spearman), ("per_sev", level, kind), or
# None for a blank separator / unpopulated column.
_VO_BLOCK = [
    # Error Detection (error-based)
    ("err_acc",   "Acc",            "error_detection_accuracy"),
    ("err_f1",    "F1 Score",       "error_detection_f1"),
    ("err_prec",  "Precision",      "error_detection_precision"),
    ("err_rec",   "Recall",         "error_detection_recall"),
    ("_b1",       "",               None),   # blank separator
    # Error Detection (sample-based)
    ("samp_acc",  "Acc",            "sample_error_detection_accuracy"),
    ("samp_f1",   "F1 Score",       "sample_error_detection_f1"),
    ("samp_prec", "Precision",      "sample_error_detection_precision"),
    ("samp_rec",  "Recall",         "sample_error_detection_recall"),
    ("_b2",       "",               None),   # blank separator
    # Variability (not in the result JSON — kept blank to mirror the formatted CSV)
    ("var_dist",  "Avg Dist Exercise", None),
    ("_b3",       "",               None),   # blank separator
    # Error Severity
    ("sev_acc",        "Acc",                 "overall_severity_accuracy"),
    ("sev_acc_w1",     "Acc - within 1",      "overall_severity_within_1"),
    ("sev_acc_non1",   "Acc (non-1)",         "overall_severity_accuracy_non1"),
    ("sev_acc_non1_w1","Acc (non-1) - within 1","overall_severity_within_1_non1"),
    ("sev_acc_1",   "Acc - 1",            ("per_sev", 1, "accuracy")),
    ("sev_acc_1_w1","Acc - 1 (within 1)", ("per_sev", 1, "within_1")),
    ("sev_acc_2",   "Acc - 2",            ("per_sev", 2, "accuracy")),
    ("sev_acc_2_w1","Acc - 2 (within 1)", ("per_sev", 2, "within_1")),
    ("sev_acc_3",   "Acc - 3",            ("per_sev", 3, "accuracy")),
    ("sev_acc_3_w1","Acc - 3 (within 1)", ("per_sev", 3, "within_1")),
    ("sev_acc_4",   "Acc - 4",            ("per_sev", 4, "accuracy")),
    ("sev_acc_4_w1","Acc - 4 (within 1)", ("per_sev", 4, "within_1")),
    ("sev_acc_5",   "Acc - 5",            ("per_sev", 5, "accuracy")),
    ("sev_acc_5_w1","Acc - 5 (within 1)", ("per_sev", 5, "within_1")),
    ("sev_mae",      "MAE",        ("raw", "overall_severity_mae")),       # MAE is NOT a pct → no ×100
    ("sev_mae_non1", "MAE (non-1)",("raw", "overall_severity_mae_non1")),
    ("_b4",       "",               None),   # blank separator
    # Effectiveness Score
    ("eff_acc",  "Score Acc",            "effectiveness_exact_match_rate"),
    ("eff_mae",  "Score MAE",            ("raw", "effectiveness_mae")),
    ("eff_pear", "Pearson Correlation",  ("raw", "effectiveness_correlation")),
    ("eff_spear","Spearman Correlation", ("raw", "effectiveness_spearman_correlation")),
    ("_b5",      "",                     None),   # blank separator
    # Injury Risk Score
    ("inj_acc",  "Score Acc",            "injury_risk_exact_match_rate"),
    ("inj_mae",  "Score MAE",            ("raw", "injury_risk_mae")),
    ("inj_pear", "Pearson Correlation",  ("raw", "injury_risk_correlation")),
    ("inj_spear","Spearman Correlation", ("raw", "injury_risk_spearman_correlation")),
]
# Sub-band labels (row-1 band) for the detail block, keyed to the first field of each sub-group.
_VO_BLOCK_SUBBANDS = {
    "err_acc":  "{stage}: Error Detection (error-based)",
    "samp_acc": "{stage}: Error Detection (sample-based)",
    "var_dist": "{stage}: Variability",
    "sev_acc":  "{stage}: Error Severity",
    "eff_acc":  "{stage}: Effectiveness Score",
    "inj_acc":  "{stage}: Injury Risk Score",
}


FIELDS = [
    # Row-family taxonomy (2026-07-09, see ~/.claude/EVAL_MAP.md) — FIRST column, so the taxonomy is
    # the first thing a reader sees. Computed LAST (in _rows()), from which VObs metric blocks are
    # actually populated on this row, never from the (possibly stale) display string. A row can
    # legitimately carry more than one family (e.g. a merged orphan row with both single-stage and
    # two-stage data) — semicolon-delimited when so. `unclassified` is a distinct sentinel for a row
    # whose populated-block pattern doesn't match any known VObs family (never guess).
    "family",
    # --- identity: who/what this row is (train_reasoning sits right before eval_thinking) ---
    "display", "model", "model_created", "owner", "is_baseline", "train_reasoning", "eval_thinking",
    # --- when: most-recent eval (model_created is up with identity) ---
    "last_eval_ts",
    # --- headline scores: the numbers you scan first ---
    "_spacer_identity",   # blank spacer column: identity │ ⎵ │ benchmarks
    #   general benchmarks (+ per-cell scoring method: parsable / judged / raw / rule)
    "MMMU_val", "Video_MME", "VSI_Bench", "IF_Bench",
    "_spacer_bench",   # blank spacer column: benchmarks │ ⎵ │ VO
    #   visual-obs detection/severity — kept in TWO non-comparable column sets (Audit 2026-06-22 F3):
    #   s1 = SINGLE-STAGE (model emits severity directly; populated by eval_all.sh visualobs);
    #   s2 = TWO-STAGE (a stage-2 reasoner consumes the model's stage-1 obs → severity). s2 are
    #   PLACEHOLDERS for the reasoner run — they fill from stage2_* files where present, else BLANK.
    "vo_s1_error_f1", "vo_s1_sample_f1", "vo_s1_severity_acc",
    "_spacer_two_stage",   # blank spacer: single-stage │ ⎵ │ two-stage
    "vo_s2_error_f1", "vo_s2_sample_f1", "vo_s2_severity_acc",
    #   visual-obs AGREEMENT vs HUMAN GT (single-stage obs; error_relevant.vs_gt.a.overall) —
    #   the comparable no-reasoner signal the old formatted CSV showed as its own band
    "vo_agree_errf1", "vo_agree_acc", "vo_agree_prec", "vo_agree_rec",
    "_spacer_vo",      # blank spacer column: VO │ ⎵ │ aux
    #   aux 3-modality headline (after visual-obs)
    "aux_acc_weighted_3mod",
    # --- aux per-modality / per-task detail. Video splits by source: 3D MCQA (mcqa_video_3d_2705,
    # the harder spatial-reasoning set) vs non-3D (mcqa_video_1505). Combined = aux_video_acc. ---
    "aux_video_acc", "aux_video_3d", "aux_video_non3d", "aux_text_acc", "aux_image_composite",
    "aux_image_dense_oks", "aux_image_dense_other", "aux_image_task4_acc",
    "_spacer_aux",     # blank spacer column: aux │ ⎵ │ metadata/provenance
    # --- training provenance (train_reasoning moved up to identity) ---
    "train_group_id", "train_sample_count",
    "_spacer_prov_train",   # blank spacer: training provenance │ ⎵ │ source provenance
    # --- source provenance / bookkeeping (last). Ordered to MATCH the results groups, with blank
    #     spacers between provenance groups: BENCHMARKS │ ⎵ │ VO │ ⎵ │ AUX. ---
    "bench_method", "bench_source",
    "_spacer_prov_bench",   # blank spacer: benchmark provenance │ ⎵ │ VO provenance
    "vo_s1_source", "vo_s1_eval_n", "vo_s2_source", "vo_s2_reasoner", "vo_s2_reasoner_thinking",
    "vo_s2_obs_source", "vo_s2_eval_n", "vo_test_set", "vo_agree_eval_n",
    "_spacer_prov_vo",      # blank spacer: VO provenance │ ⎵ │ AUX provenance
    "aux_run_ts", "aux_run_id", "aux_run_dir", "aux_source",
    # --- FULL VO METRIC DETAIL (appended AFTER all metadata; mirrors the formatted_1105 CSV's
    #     per-band metric block, for single-stage then two-stage). Repeats the headline cols on
    #     purpose so each block is a self-contained mirror. s2 mostly blank until the reasoner sweep.
    #     ONE blank spacer separates metadata from the detail block; each sub-band's first column
    #     carries its band label (stage is in the "single-stage:"/"two-stage:" prefix) — no dead
    #     label-only spacer columns. ---
    "_spacer_detail",
    *(f"vo_s1_blk_{suffix}" for suffix, _h, _src in _VO_BLOCK),
    *(f"vo_s2_blk_{suffix}" for suffix, _h, _src in _VO_BLOCK),
    # --- VO USAGE (grounding proxy, 2026-07-07) — did the stage-2 reasoner's trace actually
    #     reference the VO answers it was fed? Computed per stage2_* file from its own
    #     per_sample_results via measure_stage2_vo_usage.py (heuristic matcher; absolute rates are
    #     INFLATED — the shuffled control quantifies the floor, so read HEADROOM = mean − control,
    #     not the raw mean). Blank for single-stage rows / files with no parseable VO block.
    #     See ~/.claude/reports/visual_observations/2026-07-07_stage2_vo_usage_metric.md ---
    "_spacer_vo_usage",
    "vo_s2_usage_mean", "vo_s2_usage_full_pct", "vo_s2_usage_zero_pct",
    "vo_s2_usage_ctrl", "vo_s2_usage_headroom",
]

# Human-readable header labels for the CSV (so a paste into Excel reads cleanly). Internal field
# KEYS stay snake_case everywhere in code; only the written header row is prettified. Any field
# not listed falls back to its key with '_'->' ' and title-cased.
HEADER_LABELS = {
    "display": "Display", "model": "Model Path", "model_created": "Model Created",
    "owner": "Owner", "is_baseline": "Baseline?", "train_reasoning": "Trained w/ Reasoning",
    "eval_thinking": "Eval Thinking", "last_eval_ts": "Last Eval", "family": "Family",
    "MMMU_val": "MMMU-val", "Video_MME": "Video-MME", "VSI_Bench": "VSI-Bench",
    "IF_Bench": "IF-Bench",
    "bench_method": "Benchmark Scoring",
    # "single-stage" = model emits severity DIRECTLY in one call (no obs step); spelled out (not
    # "1-stage") so it can't be misread as "the stage-1 obs step" — single-stage SKIPS stage-1 obs.
    # NB: single-stage/two-stage/vs-GT are carried by the BAND row above — don't repeat in the label.
    "vo_s1_error_f1": "VO Error-F1", "vo_s1_sample_f1": "VO Sample-F1",
    "vo_s1_severity_acc": "VO Severity Acc",
    # "two-stage" = stage-1 obs -> stage-2 reasoner -> severity.
    "vo_s2_error_f1": "VO Error-F1", "vo_s2_sample_f1": "VO Sample-F1",
    "vo_s2_severity_acc": "VO Severity Acc",
    "vo_agree_errf1": "VO Agree-F1", "vo_agree_acc": "VO Agree-Acc",
    "vo_agree_prec": "VO Agree-Prec", "vo_agree_rec": "VO Agree-Rec",
    "aux_acc_weighted_3mod": "Aux 3-Mod Weighted",
    "aux_video_acc": "Aux Video (all)", "aux_video_3d": "Aux Video 3D", "aux_video_non3d": "Aux Video non-3D",
    "aux_text_acc": "Aux Text", "aux_image_composite": "Aux Image",
    "aux_image_dense_oks": "Aux Image Dense OKS", "aux_image_dense_other": "Aux Image Dense (non-OKS)",
    "aux_image_task4_acc": "Aux Image Task4",
    "train_group_id": "Train Group", "train_sample_count": "Train Samples",
    "aux_run_ts": "Aux Run TS", "aux_run_id": "Aux Run ID", "aux_run_dir": "Aux Run Dir",
    "aux_source": "Aux Source", "bench_source": "Benchmark Source",
    "vo_s1_source": "VO Source (single-stage)", "vo_s1_eval_n": "VO Eval N (single-stage, eval/failed)",
    "vo_s2_source": "VO Source (two-stage)",
    "vo_s2_reasoner": "Stage2 Reasoner", "vo_s2_reasoner_thinking": "Stage2 Reasoner Thinking",
    "vo_s2_obs_source": "Stage2 Obs Source",
    "vo_s2_eval_n": "VO Eval N (two-stage, eval/failed)", "vo_test_set": "VO Test Set",
    "vo_agree_eval_n": "VO Eval N (agreement, matched slots)",
    # blank spacer columns between metric groups (identity │ benchmarks │ VO │ aux │ metadata)
    "_spacer_identity": "", "_spacer_bench": "", "_spacer_vo": "", "_spacer_aux": "",
    "_spacer_prov_bench": "", "_spacer_prov_vo": "", "_spacer_prov_train": "",
    "_spacer_detail": "", "_spacer_two_stage": "", "_spacer_vo_usage": "",
    "vo_s2_usage_mean": "VO Usage Mean %", "vo_s2_usage_full_pct": "VO Usage 100% Samples %",
    "vo_s2_usage_zero_pct": "VO Usage 0% Samples %",
    "vo_s2_usage_ctrl": "VO Usage Shuffled Ctrl %", "vo_s2_usage_headroom": "VO Usage Headroom (pp)",
}
# Detail-block headers: the formatted-CSV column NAME (Acc/F1 Score/Precision/...) for each block
# field, both stages. The stage is carried by the row-1 band (see GROUP_BANDS), so the column label
# itself stays the bare formatted-CSV name (matching that CSV's disposition: repeated Acc/F1/etc.).
for _pfx in ("vo_s1", "vo_s2"):
    for _suffix, _hdr, _src in _VO_BLOCK:
        HEADER_LABELS[f"{_pfx}_blk_{_suffix}"] = _hdr


def _header(field: str) -> str:
    return HEADER_LABELS.get(field, field.replace("_", " ").title())


# Top "band" header (row 1 of a 2-row header): a group label sits on the FIRST field of each band
# and is blank for the rest, so in a spreadsheet it reads as a header spanning that band's columns.
# Makes the VO grouping unmistakable — "Single-stage" / "Two-stage" / "Agreement vs human GT" are
# DIFFERENT pipelines, not comparable. Fields not listed get a blank band cell.
GROUP_BANDS = {
    "MMMU_val": "General benchmarks",
    "vo_s1_error_f1": "Visual-obs: SINGLE-STAGE (model emits severity directly)",
    "vo_s2_error_f1": "Visual-obs: TWO-STAGE (stage-1 obs -> stage-2 reasoner)",
    "vo_agree_errf1": "Visual-obs: AGREEMENT vs human GT (stage-1 obs + rules)",
    "aux_acc_weighted_3mod": "Aux tasks",
    # NOTE: there is only ONE detail-block spacer field (`_spacer_detail`, see FIELDS) — the
    # per-stage sub-band labels below already say "single-stage:"/"two-stage:" so no separate
    # "FULL VO DETAIL (single-stage)"/"(two-stage)" banner is needed (a pair of such keys existed
    # here pointing at fields ("_spacer_detail_s1"/"_spacer_detail_s2") that were never in FIELDS —
    # dead code, removed 2026-07-06 audit fix, no information was lost).
}
for _pfx, _stage in (("vo_s1", "single-stage"), ("vo_s2", "two-stage")):
    for _suffix, _subband in _VO_BLOCK_SUBBANDS.items():
        GROUP_BANDS[f"{_pfx}_blk_{_suffix}"] = _subband.format(stage=_stage)
GROUP_BANDS["vo_s2_usage_mean"] = (
    "two-stage: VO usage (grounding proxy — heuristic matcher, read Headroom not the raw Mean)"
)


def _band_row(fields):
    """Row-1 band labels keyed to the first field of each band (blank elsewhere)."""
    return [GROUP_BANDS.get(k, "") for k in fields]


def _vo_block_fields(pfx):
    """Field keys for one stage's detail block, e.g. vo_s1_blk_err_f1."""
    return [f"{pfx}_blk_{suffix}" for suffix, _h, _src in _VO_BLOCK]


def _write_vo_block(r, m, pfx):
    """Populate one stage's full detail block from a metrics dict m into row r."""
    per_sev = m.get("per_severity_level", {}) or {}
    def _pct(v):
        return round(v * 100, 2) if isinstance(v, (int, float)) else ""
    def _raw(v):
        return round(v, 4) if isinstance(v, (int, float)) else ""
    for suffix, _h, src in _VO_BLOCK:
        key = f"{pfx}_blk_{suffix}"
        if src is None:
            continue  # blank separator / unpopulated (e.g. Variability) — leave empty
        if isinstance(src, str):
            r[key] = _pct(m.get(src))
        elif src[0] == "raw":
            r[key] = _raw(m.get(src[1]))
        elif src[0] == "per_sev":
            _, level, kind = src
            entry = per_sev.get(str(level)) or per_sev.get(level) or {}
            r[key] = _pct(entry.get(kind))


def _base_model(model: str, display: str = "") -> str:
    """Detect base model family from the served path/display. Returns '4b'|'9b'|'27b'|'other'.
    Order matters: check 27b/9b before 4b so '...-27b-...' can't be misread. Distinct 'other'
    sentinel (never default to a family) so an unrecognized model lands in its own file, not
    silently in 4b."""
    s = f"{model} {display}".lower()
    for tag in ("35-27b", "-27b", "_27b", "3.5-27b", "qwen3.5-27b"):
        if tag in s:
            return "27b"
    for tag in ("35-9b", "-9b", "_9b", "3.5-9b", "qwen3.5-9b"):
        if tag in s:
            return "9b"
    for tag in ("35-4b", "-4b", "_4b", "3.5-4b", "qwen3.5-4b"):
        if tag in s:
            return "4b"
    return "other"


def _owner(model: str) -> str:
    """Derive who owns the checkpoint from its path, so EXTERNAL (colleague) models are
    trackable without naming discipline. /sgsilva/ -> sgsilva (mine); any other /<user>/ under
    /mnt/data/ -> '<user> (external)'; HF hub ids / shared models -> '' (unknown)."""
    m = (model or "").lower()
    if "/sgsilva/" in m or "/merged_models/" in m:
        return "sgsilva"
    import re as _re
    hit = _re.search(r"/mnt/data/([a-z0-9_]+)/", m)
    if hit and hit.group(1) not in ("shared",):
        return f"{hit.group(1)} (external)"
    return ""


def _is_baseline(model: str, display: str = "") -> bool:
    """A baseline = the raw, un-SFT'd Qwen3.5 model (the reference line to sort to the top).
    Recognized by the shared-models path or a bare 'Qwen3.5-NB' id, OR a display/run_id whose
    name marks it as a baseline. SFT/GRPO/merged checkpoints are NOT baselines."""
    s = f"{model} {display}".lower()
    if "/shared/models/qwen3.5" in s:
        return True
    leaf = model.rstrip("/").split("/")[-1].lower()
    if leaf in ("qwen3.5-4b", "qwen3.5-9b", "qwen3.5-27b") or leaf.startswith("qwen3.5-") and leaf.count("-") == 1:
        return True
    if "baseline" in display.lower() or "baseline" in leaf:
        return True
    return False


def _load_allowlist():
    """Read the curated allowlist JSON ({models: [{pattern, display?, train_reasoning?, note?}]}).
    Returns a list of (pattern_lc, display, train_reasoning) tuples, or None if the file is ABSENT
    (→ keep ALL rows, legacy). A malformed file is a hard error (distinct from absent — don't
    silently fall back to keeping everything)."""
    if not MODEL_ALLOWLIST.exists():
        return None
    data = json.loads(MODEL_ALLOWLIST.read_text())
    out = []
    for idx, e in enumerate(data.get("models", [])):
        pat = (e.get("pattern") or "").strip().lower()
        if pat:
            out.append({
                "pattern": pat,
                "display": (e.get("display") or "").strip(),
                "train_reasoning": (e.get("train_reasoning") or "").strip(),
                "train_sample_count": str(e.get("train_sample_count") or "").strip(),  # curated (eval_matrix unwired)
                "group": (e.get("group") or "").strip(),
                "order": idx,  # board rows appear in allowlist order; blank row between groups
                # optional: keep ONLY this thinking-mode row for this model ("on"/"off"). Used to drop a
                # degenerate contrast row (e.g. a reasoning model served thinkoff collapses) — keep the on row.
                "keep_thinking": (e.get("keep_thinking") or "").strip().lower(),
                # optional: a labeled section banner row is written before this entry's group when
                # `section` differs from the previous kept entry's section (2026-07-06). Distinct
                # from `group` (which only controls the blank-row spacer) — a section groups several
                # `group`s under one titled row, e.g. "VOBs experiments" spanning vobs2906_bakeoff +
                # vobs2906_reasoning. Blank on most entries (no section = no banner).
                "section": (e.get("section") or "").strip(),
            })
    # Both _match_allow and the VO map are FIRST-MATCH substring maps, so a BROAD pattern listed
    # before a more SPECIFIC one would steal the specific row's curated display/train_reasoning. That
    # ordering invariant is invisible and a future append can break it — so warn when one pattern is a
    # substring of another (the order-sensitive case). Not fatal: hand-ordering can be intentional.
    for i, a in enumerate(out):
        for j, b in enumerate(out):
            if i != j and a["pattern"] in b["pattern"] and a["pattern"] != b["pattern"]:
                rel = "BEFORE" if i < j else "after"
                print(f"[allowlist WARN] pattern {a['pattern']!r} is a substring of {b['pattern']!r} "
                      f"and is listed {rel} it — first-match may mis-assign. Order specific-before-broad.")
    return out


def _vo_map_from_config():
    """Build the VO-filename→served-path map + exclude list from master_models.json — the SINGLE
    source of truth for per-model definitions (no hardcoded model names in this script). Each model
    entry contributes its `vo_tokens` (ordered filename anchors) → its `vo_path`; the order is the
    allowlist order (specific-before-broad is the file author's responsibility). Top-level
    `vo_exclude` lists probe/variant tokens that reject a file even if a token matches. Returns
    (VO_FILE_TO_MODEL: list[(token_lc, path)], VO_EXCLUDE: tuple[str]). Empty if the file is absent."""
    if not MODEL_ALLOWLIST.exists():
        return [], ()
    data = json.loads(MODEL_ALLOWLIST.read_text())
    pairs = []
    for idx, e in enumerate(data.get("models", [])):
        vp = (e.get("vo_path") or "").strip()
        if not vp:
            continue
        for j, tok in enumerate(e.get("vo_tokens") or []):
            tok = (tok or "").strip().lower()
            if tok:
                pairs.append((tok, vp, idx, j))
    # PRECEDENCE: allowlist order is BOARD-DISPLAY order (baselines first), which is NOT the
    # VO-token precedence we need — a BROAD baseline token like 'qwen3.5-27b' must resolve AFTER the
    # specific SFT tokens, else it shadows them (first-match wins). So sort by token LENGTH descending
    # (longer = more specific); ties keep allowlist order. This makes 'oracle-obs-cat-union5-step339'
    # (28 ch) beat 'qwen3.5-27b' (11 ch) without any manual ordering in the JSON.
    pairs.sort(key=lambda t: (-len(t[0]), t[2], t[3]))
    out = [(tok, vp) for tok, vp, _i, _j in pairs]
    excl = tuple((x or "").strip().lower() for x in data.get("vo_exclude", []) if (x or "").strip())
    return out, excl


def _match_allow(row: dict, allow):
    """Return the FIRST matching allowlist entry dict for a row, or None.
    Match is a substring of the row's model path OR display."""
    hay = f"{row.get('model','')} {row.get('display','')}".lower()
    for entry in allow:
        if entry["pattern"] in hay:
            return entry
    return None


# ---------------------------------------------------------------------------
# VO ROUTING — the single source of truth for "which board row does a VO result
# file land on" (stabilization step 2, 2026-07-10; plan:
# ~/.claude/reports/infra_tooling/2026-07-10_eval_pipeline_stabilization_proposal.md).
# Both the compile pass in _rows() AND the `--route` dry-run call resolve_vo(), so a
# pre-launch routing simulation can never diverge from what the compiler actually does.
# This is a FAITHFUL extraction of the logic that previously lived inline in the
# two-stage/single-stage and agreement blocks — any behavior change here must keep
# tests/test_routing.py green AND a recompiled board byte-identical, unless deliberate.
# ---------------------------------------------------------------------------

# Cohort tag (1105/1806) is recognized ONLY immediately before a known suffix — NOT a
# bare substring test. A bare "_1105" also matches a model's TRAINING-DATA tag baked
# into its checkpoint name (e.g. "oracle_obs_cat_reasoning_1105_step336_singlestage_*"),
# which would silently detach that model's VO cells onto a phantom cohort row.
_VO_COHORT_RE = re.compile(
    r"_(1105|1806)(?=_think|_gtobsbuild|_gtobs_|_modelobs|_selfloop|_singlestage|\.json$)")

_VO_CONFIG_CACHE = None


def _vo_config_cached():
    """(vo_map, vo_exclude) from master_models.json, loaded once per process."""
    global _VO_CONFIG_CACHE
    if _VO_CONFIG_CACHE is None:
        _VO_CONFIG_CACHE = _vo_map_from_config()
    return _VO_CONFIG_CACHE


def vo_cohort(fname: str) -> str:
    m = _VO_COHORT_RE.search(fname)
    return m.group(1) if m else ""


def vo_model_path(name: str, vo_map=None, vo_exclude=None) -> str:
    """Resolve a VO JSON filename -> served checkpoint path via the curated map. '' if no
    match OR an excluded probe (distinct sentinel — the caller then skips, never joins to
    a wrong key)."""
    if vo_map is None or vo_exclude is None:
        vo_map, vo_exclude = _vo_config_cached()
    if any(x in name for x in vo_exclude):
        return ""
    for tok, path in vo_map:
        if tok in name:
            return path
    return ""


def resolve_vo(name: str, metadata: dict | None = None,
               vo_map=None, vo_exclude=None, card: dict | None = None) -> dict:
    """Route one VO result filename (basename, any case) to its board row.

    card: the run-card sidecar (`<file>.card.json`, stabilization step 4) when present —
    identity fields (checkpoint_path, axis, cohort, arm, thinking, expected_n) come from
    the card, written by the PRODUCER at generation time, and filename parsing is used
    only for the tier (_v2/_cat). A carded file needs NO vo_tokens entry to reach the
    board. Card-era floors = floor(0.99 * expected_n) — reproduces the hand-set values
    (2157→2135, 2260→2237, 1181→1169). The sft2812 reasoner filter is SKIPPED for carded
    files: the card's `reasoner` field records who reasoned, and a deliberately-written
    card is trusted the way a curated vo_token is.

    metadata: the file's metadata dict when available. Admission checks that need it
    (sft2812 reasoner filter, evaluated_samples floors) run only when it is provided;
    with metadata=None (a PLANNED filename in --route) routing is still fully computed
    and `admit` stays True with skip_reason ''.

    Returns a dict:
      kind          'two_stage' | 'single_stage' | 'agreement' | None (not a routed VO file)
      excluded_by   the vo_exclude token that rejected it ('' if none)
      model_path    served checkpoint path ('' = no curated token -> INVISIBLE on the board)
      cohort        '1105' | '1806' | ''
      arm           '' | 'modelobs' | 'selfloop' | 'agree'  ('agree' marks the axis only —
                    it gets NO row-key suffix; Sandra correction 2026-07-09: __arm_* exists
                    ONLY to separate different stage-2 SETUPS, never axes of one setup)
      thinking      'on' | 'off' | 'unknown'  (a two-stage file with no think token is
                    'off' for JOIN purposes — the stage-2 reasoner is ALWAYS thinkoff,
                    decision 2026-06-04)
      tier          within-pipeline file-preference rank (_v2 beats v1; scored families
                    also prefer the _cat variant)
      row_path      model_path + optional __cohort_/__arm_ pseudo-path suffixes
      row_key       (_norm_path(row_path), thinking) — THE board row identity
      floor         min evaluated_samples for admission (None when kind=None/excluded)
      admit         False when a metadata-dependent gate (or exclusion/no-token) rejects it
      skip_reason   distinct sentinel: 'excluded:<tok>' | 'no_curated_token' |
                    'reasoner_not_sft2812' | 'below_floor:<n><<floor>' |
                    's1_below_floor:<n><<floor>' | ''
      is_expb_gtobs / is_expb_modelobs / is_expb_selfloop / is_397b_ceiling /
      is_cohort_bakeoff   the admission-exception flags (also drive obs_source)
      obs_source_hint     'GT' | 'self' | '' (two-stage naming-convention cases only)
      family_hint         EVAL_MAP family name implied by routing alone
    """
    if vo_map is None or vo_exclude is None:
        vo_map, vo_exclude = _vo_config_cached()
    name = name.lower()
    rt = {"kind": None, "excluded_by": "", "model_path": "", "cohort": "", "arm": "",
          "thinking": "unknown", "tier": 0, "row_path": "", "row_key": None,
          "floor": None, "admit": True, "skip_reason": "",
          "is_expb_gtobs": False, "is_expb_modelobs": False, "is_expb_selfloop": False,
          "is_397b_ceiling": False, "is_cohort_bakeoff": False,
          "obs_source_hint": "", "family_hint": "", "carded": False}
    if card:
        return _resolve_vo_from_card(name, card, metadata, rt)
    if name.startswith("agreement_"):
        kind = "agreement"
    elif name.startswith("stage2_"):
        kind = "two_stage"
    elif "singlestage" in name:
        kind = "single_stage"
    else:
        return rt  # obs_/judge_/other files are not routed to scored board rows
    rt["kind"] = kind
    excl = next((x for x in vo_exclude if x in name), "")
    if excl:
        rt.update(excluded_by=excl, admit=False, skip_reason=f"excluded:{excl}")
        return rt
    model_path = vo_model_path(name, vo_map, vo_exclude)
    # Fallback to metadata.model for ANY single-stage file the curated map missed: a
    # single-stage VO json records the REAL served path there (unlike two-stage, which
    # records the REASONER — the filename is the only truth for those; and agreement
    # jsons carry no metadata.model at all).
    if not model_path and kind == "single_stage" and metadata is not None:
        model_path = str(metadata.get("model", "")) or ""
    rt["model_path"] = model_path
    if not model_path:
        rt.update(admit=False, skip_reason="no_curated_token")
    if "thinkon" in name:
        thinking = "on"
    elif "thinkoff" in name:
        thinking = "off"
    else:
        thinking = "off" if kind == "two_stage" else "unknown"
    rt["thinking"] = thinking
    if kind == "agreement":
        rt["tier"] = 1 if name.endswith("_v2.json") else 0
    else:
        # _v2 rescored beats v1 (x10); the CATEGORICAL variant beats angle/other (+1) —
        # cat is the V2-canonical visual-obs variant, so a baseline with both takes _cat_
        # deterministically, not by alphabetical luck (user 2026-06-22).
        rt["tier"] = (10 if name.endswith("_v2.json") else 0) + (1 if "_cat" in name else 0)
    cohort = vo_cohort(name)
    rt["cohort"] = cohort
    # Arm/exception flags (all cohort-gated; see the EXP-B + 397B-ceiling + bake-off
    # comments formerly inline in _rows(), and EVAL_MAP.md for the family taxonomy).
    _expb = bool(cohort) and "expb" in name
    rt["is_expb_gtobs"] = _expb and "gtobsbuild" in name
    rt["is_expb_modelobs"] = _expb and "modelobs" in name
    rt["is_expb_selfloop"] = _expb and "selfloop" in name
    rt["is_397b_ceiling"] = (bool(cohort) and "397b" in name
                             and ("_gtobs_" in name or "gtobsbuild" in name)
                             and "expb" not in name)
    rt["is_cohort_bakeoff"] = bool(cohort) and "vobs2906" in name and kind == "two_stage"
    arm = ""
    if kind in ("two_stage", "agreement"):
        if rt["is_expb_modelobs"]:
            arm = "modelobs"
        elif rt["is_expb_selfloop"]:
            arm = "selfloop"
        elif kind == "agreement" and _expb and "_agree_" in name:
            arm = "agree"  # axis marker ONLY — no row-key suffix (2026-07-09 correction)
    rt["arm"] = arm
    row_path = f"{model_path}__cohort_{cohort}" if cohort else model_path
    if arm in ("modelobs", "selfloop"):
        row_path = row_path + f"__arm_{arm}"
    rt["row_path"] = row_path
    rt["row_key"] = (_norm_path(row_path), thinking)
    # Floors: min evaluated_samples for board admission. Any gtobsbuild file runs on the
    # test-BUILD's obs-covered subset (N=2157 = 2260 - 103 no-GT-obs reps) -> 2135 (99%).
    # A non-gtobsbuild cohort-tagged bake-off / 397B-ceiling file runs the FULL cohort ->
    # 2237 (99% of 2260, 1806) / 1169 (1105). Everything else keeps the legacy 1170.
    # (NOTE preserved from the inline code: EXP-B modelobs/selfloop fall through to 1170
    # even though the comment history says they run the 2157 subset — behavior kept
    # byte-identical; revisit deliberately, with tests, if that's ever tightened.)
    if kind == "two_stage":
        rt["floor"] = (2135 if "gtobsbuild" in name else
                       (2237 if cohort == "1806" else 1169)
                       if (rt["is_cohort_bakeoff"] or rt["is_397b_ceiling"]) else 1170)
    else:
        rt["floor"] = 2237 if cohort == "1806" else (1169 if cohort == "1105" else 1170)
    # Metadata-dependent admission gates (compile pass only; --route on a planned name
    # skips these by passing metadata=None).
    if metadata is not None and rt["admit"]:
        if kind == "two_stage":
            # sft2812-only board rule + the three naming-convention exceptions
            # (cohort bake-off / EXP-B native arms / 397B GT-obs ceiling).
            _rsnr = str(metadata.get("model", "")).lower()
            _is_expb_native = (rt["is_expb_gtobs"] or rt["is_expb_modelobs"]
                               or rt["is_expb_selfloop"])
            if (not (rt["is_cohort_bakeoff"] or _is_expb_native or rt["is_397b_ceiling"])
                    and not ("fe_comparison" in _rsnr and "step_2812" in _rsnr)):
                rt.update(admit=False, skip_reason="reasoner_not_sft2812")
            else:
                _eN = metadata.get("evaluated_samples") or 0
                if _eN < rt["floor"]:
                    rt.update(admit=False, skip_reason=f"below_floor:{_eN}<{rt['floor']}")
        elif kind == "single_stage":
            _eN = metadata.get("evaluated_samples") or 0
            if _eN < rt["floor"]:
                rt.update(admit=False, skip_reason=f"s1_below_floor:{_eN}<{rt['floor']}")
    # Naming-convention obs-source + family (routing-implied; the compile pass may refine
    # obs_source from metadata for the plain FIXED-REASONER case).
    if kind == "two_stage":
        if rt["is_expb_gtobs"] or rt["is_397b_ceiling"]:
            rt["obs_source_hint"] = "GT"
            rt["family_hint"] = "2-stage-GT-VObs"
        elif rt["is_expb_selfloop"]:
            rt["obs_source_hint"] = "self"
            rt["family_hint"] = "2-stage-OWN-MODEL-VObs"
        elif rt["is_expb_modelobs"]:
            rt["family_hint"] = "2-stage-DIF-MODEL-VObs"
        else:
            rt["family_hint"] = "2-stage-FIXED-REASONER-VObs"
    elif kind == "single_stage":
        rt["family_hint"] = "Single-stage"
    else:
        rt["family_hint"] = "VObs-agreement"
    return rt


# V2 ERA TESTSET POLICY: the aux axis MUST be the new testset_1506 for every board model (the
# only exception is the standalone reduced3 comparison CSV, which is built separately to SHOW the
# testset effect). The testset isn't a structured field anywhere — it's only inferable from the
# run_id / tag / eval_family naming. So classify, and accept ONLY '1506'. Anything we can't
# confidently classify is 'unknown' and is EXCLUDED (never silently passed as 1506).
# V2 era boundary: the eval_all.sh pipeline (TESTSET=1506 by default) became the standard on
# 2026-06-17. An aux run STAMPED on/after this whose naming carries no OLD-testset marker is a
# V2 (1506) run — this is how we accept new-pipeline runs (B grpo492, A sft2812, the fresh
# baselines) that don't spell '1506' in their run_id.
_V2_TS_BOUNDARY = "2026-06-17"
_OLD_TESTSET_MARKERS = ("reduced3", "reduced2", "test_reduced", "1405", "2605", "2403",
                        "20260331", "20260401", "20260406", "20260407", "20260408",
                        "indomain", "skeleton")
def _aux_testset(run_id: str, eval_family: str = "", tag: str = "", timestamp: str = "") -> str:
    """Classify an aux run's testset. Returns '1506' | 'old' | 'unknown'.
    Rules (in order): explicit '1506' token -> 1506; explicit OLD marker / legacy 'baseline'
    family -> old; otherwise a run stamped on/after the V2 boundary -> 1506 (new-pipeline default);
    else unknown (EXCLUDED — never silently treated as 1506)."""
    s = f"{run_id} {eval_family} {tag}".lower()
    if "1506" in s:
        return "1506"
    if any(t in s for t in _OLD_TESTSET_MARKERS):
        return "old"
    if eval_family.strip().lower() == "baseline":
        return "old"
    # DATE FALLBACK (allow-by-absence): a run stamped on/after the V2 boundary with NO explicit 1506
    # token and NO old-marker is ASSUMED 1506. This is the one place we allow-by-absence instead of
    # excluding the unknown — so it's a distinct sentinel '1506?' that the caller (a) treats as 1506
    # but (b) LOGS, so a new experiment on yet-another testset stamped after the boundary is visible,
    # not silently passed. _OLD_TESTSET_MARKERS must stay exhaustive for this to be safe.
    ts = (timestamp or "").strip()
    if ts:
        # the comparison is lexicographic and only valid for ISO YYYY-MM-DD. Guard against epoch
        # seconds / other formats sneaking in (would make `>=` garbage).
        if len(ts) >= 10 and ts[4] == "-" and ts[7] == "-" and ts[:4].isdigit():
            if ts[:10] >= _V2_TS_BOUNDARY:
                return "1506?"   # date-fallback → caller treats as 1506 AND logs
        else:
            return "unknown"     # non-ISO timestamp → can't date-classify, exclude (don't guess)
    return "unknown"


def _write_csv(path: Path, row_items, fields):
    """Write rows. If rows carry an allowlist `_order` (curated board), emit in THAT order and
    insert a BLANK row between groups (`_group` change) — the user-defined layout. Otherwise
    (legacy / no allowlist) pin baselines to the top, then alphabetical by model."""
    have_order = any("_order" in r for _k, r in row_items)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        raw = csv.writer(f)
        raw.writerow(_band_row(fields))               # row 1: band labels (spanning groups)
        raw.writerow([_header(k) for k in fields])    # row 2: per-column metric labels
        if have_order:
            # within an allowlist position, keep a stable secondary sort (thinkoff before thinkon)
            _th = {"off": 0, "on": 1, "unknown": 2}
            ordered = sorted(row_items, key=lambda kv: (kv[1].get("_order", 1e9),
                                                        _th.get(kv[1].get("eval_thinking", ""), 3)))
            n, prev_group, prev_section = 0, None, None
            for _key, row in ordered:
                g = row.get("_group", "")
                sec = row.get("_section", "")
                if prev_group is not None and g != prev_group:
                    w.writerow({k: "" for k in fields})  # blank separator row between groups
                # SECTION BANNER (2026-07-06): a titled row spanning several `group`s, written once
                # right after that blank separator when `_section` changes into a non-empty value —
                # e.g. "VOBs experiments" ahead of the vobs2906 rows. `display` carries the title;
                # every other field stays blank so it reads as a header, not a data row.
                if sec and sec != prev_section:
                    banner = {k: "" for k in fields}
                    banner["display"] = sec
                    w.writerow(banner)
                w.writerow({k: row.get(k, "") for k in fields})
                prev_group = g
                prev_section = sec
                n += 1
            return n
        ordered = sorted(row_items, key=lambda kv: (0 if _is_baseline(kv[1].get("model", ""), kv[1].get("display", "")) else 1, kv[0]))
        for _key, row in ordered:
            w.writerow({k: row.get(k, "") for k in fields})
        return len(ordered)


# A bare baseline is scored on different axes under DIFFERENT identity strings (aux uses
# 'Qwen3.5-4b', VO uses the local shared path, benchmarks use the 'Qwen/Qwen3.5-4B' hub id) —
# so without normalization the SAME baseline splits into 3 rows. Map every known alias (lowercased,
# trailing-slash-stripped) -> one canonical served path so the axes join into a single row.
_BASELINE_ALIASES = {
    "qwen3.5-4b": "/mnt/data/shared/models/Qwen3.5-4B",
    "qwen/qwen3.5-4b": "/mnt/data/shared/models/Qwen3.5-4B",
    "/mnt/data/shared/models/qwen3.5-4b": "/mnt/data/shared/models/Qwen3.5-4B",
    "qwen3.5-27b": "/mnt/data/shared/models/Qwen3.5-27B",
    "qwen/qwen3.5-27b": "/mnt/data/shared/models/Qwen3.5-27B",
    "/mnt/data/shared/models/qwen3.5-27b": "/mnt/data/shared/models/Qwen3.5-27B",
    "qwen3.5-397b-a17b": "/mnt/data/shared/models/Qwen3.5-397B-A17B",
    "/mnt/data/shared/models/qwen3.5-397b-a17b": "/mnt/data/shared/models/Qwen3.5-397B-A17B",
}


def _norm_path(p: str) -> str:
    """Canonical join key = the served checkpoint path, normalized (strip trailing slash,
    resolve symlinks where possible). All three pipelines serve the SAME path, so this is the
    one reliable key. Bare-baseline aliases collapse to one canonical path (see _BASELINE_ALIASES).
    Falls back to the raw string when not a real path (e.g. an unknown HF hub id)."""
    if not p:
        return ""
    s = str(p).rstrip("/")
    canon = _BASELINE_ALIASES.get(s.lower())
    if canon:
        return canon
    try:
        rp = os.path.realpath(s)
        if os.path.exists(rp):
            return rp.rstrip("/")
    except Exception:
        pass
    return s


def _fmt_ts(epoch) -> str:
    """epoch seconds -> 'YYYY-MM-DD HH:MM' (local), or '' on failure. Distinct empty sentinel
    so a missing timestamp can't masquerade as a real one."""
    if not epoch:
        return ""
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _model_created(model_path: str) -> str:
    """When the checkpoint was CREATED = mtime of the served checkpoint dir/file. For an HF
    export dir that's the dir mtime (set at export); we also peek config.json (the file written
    at export) for a tighter stamp. '' if the path doesn't exist (e.g. HF hub id)."""
    p = _norm_path(model_path)
    if not p or not os.path.exists(p):
        return ""
    try:
        best = os.path.getmtime(p)
        cfg = os.path.join(p, "config.json")
        if os.path.isfile(cfg):
            best = max(best, os.path.getmtime(cfg))
        return _fmt_ts(best)
    except Exception:
        return ""


def _newest_mtime(*paths) -> float:
    """Max mtime over the given files/dirs that exist (0 if none)."""
    best = 0.0
    for p in paths:
        if not p:
            continue
        try:
            if os.path.exists(p):
                best = max(best, os.path.getmtime(p))
        except Exception:
            pass
    return best


# a prediction VLMEvalKit couldn't get a real answer for (runaway thinkon generation that
# timed out / hit max_tokens). These are NON-RESPONSES, not wrong answers — exclude from the
# denominator so accuracy reflects only parsable answers.
_UNPARSED_MARKERS = ("failed to obtain answer", "api error", "")


def _parsable_bench_acc(disp: str, bench: str, thinking: str = "off"):
    """Recompute a benchmark's accuracy over PARSABLE answers only, from the per-sample
    *_result.xlsx VLMEvalKit writes (cols: hit, prediction). Excludes 'Failed to obtain answer'
    non-responses from BOTH numerator and denominator. Returns (pct, n_used, n_dropped) or None
    if no result file / no pandas. The raw summary.csv value remains the fallback.

    Empty-prediction handling depends on `thinking`: a thinkON run that hits max_tokens truncates
    BEFORE emitting an answer → an empty string is a genuine NON-RESPONSE (drop it). A thinkOFF run
    has no runaway to truncate, so an empty answer is plausibly a REAL (wrong) answer → KEEP it
    (dropping it would inflate accuracy). 'Failed to obtain answer' / 'api error' are always drops."""
    try:
        import pandas as pd
    except Exception:
        return None
    bdir = BENCH_RESULTS / bench / disp
    if not bdir.is_dir():
        return None
    # newest *_result.xlsx under the display dir (skip the T-timestamp subdirs' dupes by mtime)
    cands = sorted(bdir.rglob("*_result.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        cands = sorted(bdir.rglob("*_acc.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    try:
        df = pd.read_excel(cands[0])
    except Exception:
        return None
    if "hit" not in df.columns or "prediction" not in df.columns:
        return None
    pred = df["prediction"].astype(str).str.strip().str.lower()
    bad = pred.str.contains("failed to obtain answer", na=False) | pred.str.contains("api error", na=False)
    if thinking == "on":
        bad = bad | pred.eq("")  # thinkON: empty = truncated non-response → drop. thinkOFF: keep (real wrong).
    keep = df[~bad]
    if len(keep) == 0:
        return None
    return round(keep["hit"].mean() * 100, 2), len(keep), int(bad.sum())


def _bench_display_to_path() -> dict[str, str]:
    """Map a benchmark summary.csv display_name -> served model path WITHOUT needing config
    files. The path is encoded in the result tree: results/<bench>/<display>/<model_slug>/,
    where model_slug = the served path with '/' -> '--'. We decode it back. (Configs are NOT
    required for the join — this works for any benchmark result, eval_all-driven or not.)"""
    out: dict[str, str] = {}
    bench_results = BENCH_RESULTS
    for bench in ("mmmu_val", "video_mme", "vsibench", "ifbench",
                  "mmmu_val_judged", "video_mme_judged", "vsibench_judged"):
        bdir = bench_results / bench
        if not bdir.is_dir():
            continue
        for disp_dir in bdir.iterdir():
            if not disp_dir.is_dir():
                continue
            for slug_dir in disp_dir.iterdir():
                name = slug_dir.name
                if slug_dir.is_dir() and name.startswith("--"):
                    # "--mnt--data--x" -> "/mnt/data/x" (slug IS the served path; for long external
                    # ckpts this is a models/_ext/ symlink, which _norm_path resolves to the real path)
                    out.setdefault(disp_dir.name, "/" + name.lstrip("-").replace("--", "/"))
                    break
    return out


def _video_source_split(video_results_json: str):
    """Read a video eval results_json's `by_source` and return (acc_3d, acc_non3d) percentages,
    or ('', '') if unavailable. The 1506 video stage evaluates TWO sources tagged
    metadata.source_dataset: mcqa_video_3d_2705 (3D spatial MCQA, harder) and mcqa_video_1505
    (non-3D). Each by_source entry carries an `accuracy` (already a pct). Distinct '' sentinel on
    any miss — never fabricate a number."""
    if not video_results_json:
        return "", ""
    try:
        bs = (json.loads(Path(video_results_json).read_text()).get("by_source") or {})
    except Exception:
        return "", ""
    def _acc(src):
        v = bs.get(src)
        if isinstance(v, dict):
            a = v.get("accuracy")
            if isinstance(a, (int, float)):
                return round(a, 2)
        return ""
    return _acc("mcqa_video_3d_2705"), _acc("mcqa_video_1505")


def _has(r, *fields) -> bool:
    """True if ANY of the given fields is populated. A genuine 0.0 score is a REAL value (not
    'missing') — only empty string counts as unpopulated, never a magic-number exclusion."""
    return any(str(r.get(f, "")).strip() != "" for f in fields)


def _classify_family(r: dict) -> str:
    """Row-family taxonomy per ~/.claude/EVAL_MAP.md, derived ONLY from which VObs metric-block
    columns this row actually has populated (+ the `_arm`/`vo_s2_obs_source` markers already written
    by the two-stage/agreement blocks above) — never from `display`, which can be stale (the exact
    ambiguity that caused the 2026-07-09 flipfix step840 incident). Aux/Benchmarks are NOT part of
    this taxonomy — they ride along on nearly every row regardless of VObs family, and their presence
    is already directly checkable via their own populated columns (`aux_acc_weighted_3mod`, `MMMU_val`,
    etc.), so naming them here would just repeat "Aux;Benchmarks" on ~40 rows for no signal. A row can
    legitimately match more than one VObs family (e.g. a merged single-stage/two-stage orphan row, see
    step562 gtobs) — semicolon-joined, most-specific first. Returns 'unclassified' (never a guessed
    happy-path label) when a row has NO populated VObs block at all — e.g. a pure section-banner row,
    or a future block type this function doesn't yet know about.

    2026-07-09 naming (Sandra correction, same day as the initial add): the FOUR two-stage families
    all answer "whose VObs, reasoned over by whom" — do not confuse OWN-MODEL-VObs (one model, BOTH
    roles) with FIXED-REASONER-VObs (the checkpoint's own VObs, but a DIFFERENT fixed model reasons):
      - 2-stage-GT-VObs             : human-GT VObs inlined into the trained prompt (ceiling test)
      - 2-stage-OWN-MODEL-VObs      : the checkpoint generates its OWN VObs AND reasons over them
                                      itself (one model, both roles, closed loop — EVAL_MAP.md's
                                      "SELF-LOOP arm"; historically the WORST of the two-stage arms)
      - 2-stage-FIXED-REASONER-VObs : the checkpoint's own VObs, consumed by a DIFFERENT, FIXED
                                      reference reasoner (e.g. sft2812) — the norm/default two-stage
                                      setup for a plain/baseline checkpoint (EVAL_MAP.md's "MODEL-obs
                                      arm (the norm for a plain/baseline checkpoint)")
      - 2-stage-DIF-MODEL-VObs      : a DIFFERENT, already-served model's VObs feeds the reasoner
                                      under test (EXP-B's cross-model drop-in arm)
    """
    fams = []
    has_s1 = _has(r, "vo_s1_error_f1", "vo_s1_sample_f1", "vo_s1_severity_acc")
    has_s2 = _has(r, "vo_s2_error_f1", "vo_s2_sample_f1", "vo_s2_severity_acc")
    has_agree = _has(r, "vo_agree_errf1", "vo_agree_acc")
    arm = (r.get("_arm") or "").strip()
    obs_src = (r.get("vo_s2_obs_source") or "").strip()

    if has_s2:
        if arm == "modelobs":
            # EXP-B's cross-model arm: a DIFFERENT, already-served model's VObs feeds the reasoner
            # under test (a real drop-in-pipeline test) — distinct from both OWN-MODEL-VObs (the
            # checkpoint reasons over its OWN VObs) and FIXED-REASONER-VObs (the checkpoint's own
            # VObs but a DIFFERENT FIXED reasoner, e.g. sft2812, does the reasoning) below.
            fams.append("2-stage-DIF-MODEL-VObs")
        elif arm == "selfloop":
            # the checkpoint generates its OWN stage-1 VObs (untrained task, live call) then
            # consumes those SAME self-generated VObs in its own stage-2 call — ONE model plays
            # BOTH the VObs-producer and reasoner roles (a closed loop, no external help).
            fams.append("2-stage-OWN-MODEL-VObs")
        elif obs_src == "GT":
            fams.append("2-stage-GT-VObs")
        else:
            # Any other two-stage row — a resolved (non-EXP-B) obs source, OR obs-source metadata
            # missing/unrecorded (never observed in the current 65 rows, but not impossible for a
            # future run with a metadata gap) — is the ordinary/production-realistic mode: the
            # checkpoint's own VObs consumed by a DIFFERENT, FIXED reference reasoner (e.g. sft2812)
            # — see EVAL_MAP.md's "MODEL-obs arm (the norm for a plain/baseline checkpoint)" row.
            # This is the DEFAULT two-stage family (Sandra 2026-07-09: no separate "unknown" sentinel
            # — an unresolved obs source is assumed to be this common case, not flagged).
            fams.append("2-stage-FIXED-REASONER-VObs")
    if has_s1:
        fams.append("Single-stage")
    if has_agree:
        # the self-agreement EXP-B arm (checkpoint scored vs GT with no stage-2 call) is the SAME
        # "VObs agreement" family as every other agreement row — EVAL_MAP.md draws no distinction
        # (arm only matters for whose VObs are being scored, carried by vo_s2_obs_source, not by Family).
        fams.append("VObs-agreement")

    return ";".join(fams) if fams else "unclassified"


def _rows():
    """Return {(model_path, thinking): {field: value}} JOINED on the served checkpoint path."""
    rows: dict[tuple[str, str], dict] = {}
    datefallback_runs: list[str] = []  # aux runs classed 1506 ONLY by the date fallback (logged)

    def get(model_path, thinking, display=""):
        key = (_norm_path(model_path), thinking)
        r = rows.setdefault(key, {"model": key[0], "eval_thinking": thinking, "owner": _owner(key[0])})
        if display and not r.get("display"):
            r["display"] = display
        return r

    # ---- AUX (PRIMARY): the rich eval_matrix.csv — keyed on the served `model` path, carries
    # train_reasoning / train_sample_count / acc_weighted_3modalities / per-task breakdowns. We
    # promote the scalars that matter into the master (full per-task detail stays in eval_matrix).
    def _f(v):
        v = (v or "").strip()
        try:
            return round(float(v), 2)
        except (TypeError, ValueError):
            return ""
    aux_seen = set()
    aux_seen_runids = set()  # (run_id, thinking) the matrix already covers — so the fallback
                             # JSON can't re-add the SAME run under a different path-string
                             # (e.g. baseline scored as 'Qwen3.5-27b' in matrix vs
                             # 'qwen3.5-27b/baseline_20260401' in the JSON) → spurious dup rows.
    for matrix in (AUX_MATRIX_4B, AUX_MATRIX):  # per-4B first, then combined (fills 9b/27b)
        if not matrix.exists():
            continue
        with matrix.open() as f:
            for rec in csv.DictReader(f):
                model_path = (rec.get("model") or "").strip()
                if not model_path:
                    continue  # distinct sentinel: skip empty-model rows, don't invent a key
                # V2 aux MUST be testset_1506 — exclude old/unknown-testset aux runs so the board's
                # aux axis is internally comparable (the reduced3 comparison lives in its own CSV).
                _ts_class = _aux_testset(rec.get("run_id", ""), rec.get("eval_family", ""), rec.get("tag", ""), rec.get("timestamp", ""))
                if _ts_class not in ("1506", "1506?"):
                    continue
                if _ts_class == "1506?":
                    datefallback_runs.append(f"{rec.get('run_id','')} @ {rec.get('timestamp','')[:10]} [{matrix.name}]")
                thinking = "on" if "thinkon" in (rec.get("run_id", "").lower()) else (
                    "off" if "thinkoff" in (rec.get("run_id", "").lower()) else "unknown")
                k = (_norm_path(model_path), thinking)
                if k in aux_seen:  # combined matrix shouldn't override the per-base-model one
                    continue
                aux_seen.add(k)
                rid = (rec.get("run_id", "").strip().lower(), thinking)
                if rid[0]:
                    aux_seen_runids.add(rid)
                r = get(model_path, thinking, display=f"{rec.get('base_model','')}:{rec.get('run_id','')}")
                r["aux_acc_weighted_3mod"] = _f(rec.get("acc_weighted_3modalities"))
                r["aux_video_acc"] = _f(rec.get("acc_video"))
                # 3D/non-3D video split — the matrix has no per-source column, so read by_source
                # from the video run's results_json (under video_run_dir/.../results/*.json).
                _vrd = rec.get("video_run_dir", "").strip()
                if _vrd:
                    # the FINAL results json, NOT the partial *.checkpoint.json (resume sidecar);
                    # newest by mtime among non-checkpoint files.
                    _cands = [p for p in Path(_vrd).rglob("results/*.json")
                              if not p.name.endswith(".checkpoint.json")]
                    _vrj = max(_cands, key=lambda p: p.stat().st_mtime, default=None) if _cands else None
                    if _vrj:
                        _v3d, _vn3d = _video_source_split(str(_vrj))
                        if _v3d != "": r["aux_video_3d"] = _v3d
                        if _vn3d != "": r["aux_video_non3d"] = _vn3d
                r["aux_text_acc"] = _f(rec.get("acc_text"))
                r["aux_image_composite"] = _f(rec.get("acc_image"))
                r["aux_image_dense_oks"] = _f(rec.get("oks_image"))
                r["aux_image_dense_other"] = (rec.get("image_dense_other") or "").strip()  # non-OKS dense metrics (task2 ExactMatch / task3a F1)
                # task4a preferred, task4b fallback — but a genuine 0.0 on task4a is a REAL score,
                # not "missing". `_f("0.0") or _f(task4b)` would coalesce that 0 away (falsy-zero bug),
                # so test presence explicitly: use task4a iff it parsed to a number, else task4b.
                _t4a = _f(rec.get("acc_task4a"))
                r["aux_image_task4_acc"] = _t4a if _t4a != "" else _f(rec.get("acc_task4b"))
                r["train_reasoning"] = rec.get("train_reasoning", "")
                r["train_group_id"] = rec.get("train_group_id", "")
                # '0' is the eval_matrix UNSET sentinel (no model trains on 0 samples) → treat as blank,
                # so it doesn't masquerade as a real count. Real counts come curated from master_models.json.
                _tsc = rec.get("train_sample_count", "")
                r["train_sample_count"] = "" if str(_tsc).strip() in ("", "0") else _tsc
                r["aux_run_ts"] = rec.get("timestamp", "")
                r["aux_run_id"] = rec.get("run_id", "")
                r["aux_run_dir"] = rec.get("multimodal_run_dir", "")
                r["aux_source"] = matrix.name

    # ---- AUX (FALLBACK): per-run aggregate JSON for runs not yet in eval_matrix (matrix export
    # is manual, so a fresh run can lag). Only fills modalities the matrix row didn't already set.
    if AUX_EVALS.is_dir():
        for agg in AUX_EVALS.glob("*/multimodal/*/*/*/*/results/multimodal_*.json"):
            try:
                d = json.loads(agg.read_text())
            except Exception:
                continue
            run_id = str(d.get("run_id", ""))
            # V2 aux MUST be testset_1506 (same gate as the matrix source).
            _ts_class = _aux_testset(run_id, str(d.get("eval_family", "")), str(d.get("tag", "")), str(d.get("created_at", "")))
            if _ts_class not in ("1506", "1506?"):
                continue
            if _ts_class == "1506?":
                datefallback_runs.append(f"{run_id} @ {str(d.get('created_at',''))[:10]} [fallback-json]")
            tl = (run_id + " " + str(d.get("tag", ""))).lower()
            thinking = "on" if "thinkon" in tl else ("off" if "thinkoff" in tl else "unknown")
            mods = d.get("modalities", {}) or {}
            video_rj = (mods.get("video") or {}).get("results_json")  # for the 3d/non3d source split
            model_path = ""
            for leg in ("video", "text", "image"):
                rj = (mods.get(leg) or {}).get("results_json")
                if rj:
                    rmeta = Path(rj).parent.parent / "RUN_METADATA.json"
                    if rmeta.exists():
                        try:
                            model_path = json.loads(rmeta.read_text()).get("model", "")
                        except Exception:
                            pass
                    if model_path:
                        break
            if not model_path:
                model_path = f"{d.get('base_model','')}/{run_id}"  # fallback sentinel key
            if (_norm_path(model_path), thinking) in aux_seen:
                continue  # eval_matrix already covered this run (richer) — skip the thin JSON
            if (run_id.strip().lower(), thinking) in aux_seen_runids:
                continue  # SAME run_id as a matrix row but a different path-string → it's the
                          # same eval; the matrix row is authoritative, don't add a dup row
            r = get(model_path, thinking, display=str(d.get("base_model", "")) + ":" + run_id)
            def pct(mod):
                v = (mods.get(mod) or {}).get("metric_value_pct")
                return round(v, 2) if isinstance(v, (int, float)) else ""
            r["aux_video_acc"] = pct("video")
            _v3d, _vn3d = _video_source_split(video_rj)  # 3D MCQA vs non-3D split (by_source)
            if _v3d != "": r["aux_video_3d"] = _v3d
            if _vn3d != "": r["aux_video_non3d"] = _vn3d
            r["aux_text_acc"] = pct("text")
            r["aux_image_composite"] = pct("image")
            r["aux_image_dense_oks"] = pct("image_dense")
            r["aux_image_task4_acc"] = pct("image_task4")
            # Headline `acc_weighted_3modalities` = equal-weight mean(video, text, image)
            # (verified vs 40+ eval_matrix rows). The matrix source reads it from a column; the
            # fallback JSON has no such field, so compute it here — else a fresh run (not yet
            # exported to the matrix) shows a BLANK 'Aux 3-Mod Weighted' headline on the board.
            _v, _t, _i = pct("video"), pct("text"), pct("image")
            if all(isinstance(x, (int, float)) for x in (_v, _t, _i)):
                r["aux_acc_weighted_3mod"] = round((_v + _t + _i) / 3, 2)
            r["aux_run_id"] = run_id
            r["aux_run_dir"] = str(d.get("run_dir", agg.parent.parent))
            r["aux_source"] = "multimodal_json(fallback)"

    # ---- BENCHMARKS: read BOTH summary.csv (broad) then overlay summary_judge.csv
    # (judged preferred where present). NOT all-or-nothing: a judge CSV from an older
    # run must not hide models that only have raw results in the current summary.csv. ----
    disp2path = _bench_display_to_path()
    def _load_bench(bench_csv):
        if not bench_csv.exists():
            return
        with bench_csv.open() as f:
            for rec in csv.DictReader(f):
                disp = (rec.get("Model") or "").strip()
                if not disp:
                    continue
                # prefer an explicit thinkon/thinkoff token in the display; else the Reasoning col
                dl = disp.lower()
                if "thinkon" in dl:
                    thinking = "on"
                elif "thinkoff" in dl:
                    thinking = "off"
                else:
                    thinking = "on" if (rec.get("Reasoning", "").strip().lower() in ("yes", "true", "on")) else "off"
                model_path = disp2path.get(disp, disp)  # path if config known, else display
                r = get(model_path, thinking, display=disp)
                # remember the BENCHMARK display name (= the benchmark result-tree dir name). The row's
                # `display` is whatever pipeline populated it FIRST (aux runs first → `<base>:<run_id>`),
                # which is NOT the benchmark dir name, so last_eval_ts must use this, not `display`.
                r["_bench_display"] = disp
                def num(c):
                    v = (rec.get(c) or "").strip()
                    try:
                        return round(float(v) * 100, 2) if v and float(v) <= 1.0 else (round(float(v), 2) if v else "")
                    except ValueError:
                        return ""
                mmmu, vmme, vsi = num("MMMU-val"), num("Video-MME"), num("VSI-Bench")
                # IFBench can carry the distinct string sentinel "apifail" (collect_results.py,
                # 2026-07-06 audit fix P1.4) when >2% of predictions were API/infra failures rather
                # than genuine rule non-compliance — num() would silently swallow that into "" (a
                # blank cell reads as "never run", masking a real but poisoned/deflated run). Surface
                # it as its own value so the board cell is visibly "apifail", not blank.
                _ifb_raw = (rec.get("IF-Bench") or "").strip()
                ifb = "apifail" if _ifb_raw == "apifail" else num("IF-Bench")
                # which summary file is this? raw=non-responses counted wrong; judged=LLM-judged
                method = "judged" if "judge" in bench_csv.name.lower() else "raw"
                meth = dict(_kv.split("=", 1) for _kv in (r.get("bench_method") or "").split(";") if "=" in _kv)
                # only overwrite a column when this file actually has a value for it
                if mmmu != "": r["MMMU_val"] = mmmu; meth["MMMU"] = method
                if vmme != "": r["Video_MME"] = vmme; meth["VMME"] = method
                if vsi != "": r["VSI_Bench"] = vsi; meth["VSI"] = method
                # IFBench is rule-scored (deterministic checkers) — method is ALWAYS "rule",
                # never judged/parsable, and it is NEVER added to the _parsable_bench_acc loop
                # below (that loop drops non-responses, which for a rule-scored benchmark are
                # legitimate fails — dropping them would inflate the score).
                if ifb != "": r["IF_Bench"] = ifb; meth["IFB"] = "rule"
                src = r.get("bench_source") or ""
                srcs = [src, bench_csv.name]
                # OVERRIDE with parsable-only accuracy (exclude 'Failed to obtain answer' non-
                # responses from the denominator). raw summary value stays the fallback.
                # NOTE: parsable only fires where a per-sample _result.xlsx exists, so within one
                # row some benchmarks may be 'parsable' and others 'raw'/'judged' — bench_method
                # records the method PER cell so the mix is visible, not silently conflated.
                for col, bench, mlabel in (("MMMU_val", "mmmu_val", "MMMU"), ("Video_MME", "video_mme", "VMME"), ("VSI_Bench", "vsibench", "VSI")):
                    # Do NOT clobber a JUDGED cell with raw-parsable: the judge rescored
                    # right-but-unparsed \boxed{}/prose answers (the honest, higher number);
                    # _parsable_bench_acc reads the RAW per-sample xlsx and would DROP those same
                    # rescued answers as non-responses → understates the model (the inverse of the
                    # judge's purpose). Parsable only fills/overrides raw cells, never judged.
                    if meth.get(mlabel) == "judged":
                        continue
                    pa = _parsable_bench_acc(disp, bench, thinking)
                    if pa is not None and pa[2] > 0:  # only when some were actually dropped
                        r[col] = pa[0]
                        meth[mlabel] = f"parsable(-{pa[2]})"
                        srcs.append(f"{bench}:parsable({pa[1]},-{pa[2]})")
                r["bench_source"] = ",".join(dict.fromkeys(filter(None, srcs)))
                r["bench_method"] = ";".join(f"{k}={v}" for k, v in meth.items())
    _load_bench(BENCH_SUMMARY)        # raw first (broad coverage)
    _load_bench(BENCH_SUMMARY_JUDGE)  # judged overlays where present (preferred)

    # ---- VISUAL-OBS ----
    # CRITICAL JOIN NOTE: for a TWO-STAGE VO run, metadata.model is the STAGE-2 REASONER
    # (e.g. Qwen3.5-27B), NOT the stage-1 VO model under test — so EVERY two-stage run collides
    # on the reasoner path. The stage-1 JSONs carry model=None. The ONLY reliable identifier of
    # the VO model is the FILENAME. So we map VO filename tokens -> served checkpoint path
    # explicitly (curated, like master_models.txt). A `stage2_*` result is the deployable
    # two-stage number (the headline); a `*_singlestage_*` result is the no-observations variant.
    # Precedence per (model,thinking): stage2 > singlestage; *_v2 (rescored, see feedback_eval_
    # gotchas §4) > non-v2. We pick the BEST-tier file per key and never let a worse tier win.
    # Curated map: (anchor-token, served path) + the probe/variant exclude list — BOTH now sourced
    # from master_models.json (per-model `vo_tokens`→`vo_path`, top-level `vo_exclude`). The script
    # holds NO hardcoded model names; to add/rename a model edit ONLY the JSON. Order = allowlist
    # order (specific-before-broad is the file author's responsibility); first matching token wins.
    # The dashed↔underscored sibling tokens, the llmfms-probe exclude (so the clean dashed file wins,
    # Audit 2026-06-22 F4), and the bare-baseline dotted variants all live in the JSON now.
    VO_FILE_TO_MODEL, VO_EXCLUDE = _vo_map_from_config()
    # path -> curated display name, for resolving an OBS-SOURCE model's path to a readable label
    # (vo_s2_obs_source column) rather than a raw /mnt/data/... path. Built once here since it's
    # only needed by that one column, not the filename->path resolution VO_FILE_TO_MODEL does.
    _VO_PATH_TO_DISPLAY = {}
    if MODEL_ALLOWLIST.exists():
        for _e in json.loads(MODEL_ALLOWLIST.read_text()).get("models", []):
            _vp = (_e.get("vo_path") or "").strip()
            _disp = (_e.get("display") or "").strip()
            if _vp and _disp and _norm_path(_vp) not in _VO_PATH_TO_DISPLAY:
                _VO_PATH_TO_DISPLAY[_norm_path(_vp)] = _disp
    def _vo_model_path(name: str) -> str:
        """Local alias binding this run's map/exclude — used by the obs-source resolution
        below; row ROUTING itself goes through module-level resolve_vo()."""
        return vo_model_path(name, VO_FILE_TO_MODEL, VO_EXCLUDE)
    # SINGLE-STAGE vs TWO-STAGE are NOT comparable (a two-stage stage-2 reasoner consumes the model's
    # stage-1 obs → severity; single-stage emits severity directly). The board keeps BOTH in SEPARATE
    # column sets so they never silently mix (Audit 2026-06-22 F3). Each pipeline has its own _v2-vs-v1
    # tier track. Two-stage columns are a PLACEHOLDER for the user's reasoner run — they fill from
    # stage2_* files where present (e.g. the historical 397B oracle ceiling) and stay BLANK otherwise.
    # (file-preference tiers now computed by resolve_vo(): _v2 ×10 + _cat +1 for the
    #  scored families, _v2 = 1 for agreement — see its docstring)
    if VO_RUNS.is_dir():
        ss_best: dict[tuple[str, str], int] = {}  # single-stage (model,thinking) -> winning _v2 tier
        ts_best: dict[tuple[str, str], int] = {}  # two-stage   (model,thinking) -> winning _v2 tier
        # ---- COHORT-AWARE VO ROWS (2026-07-02) ----
        # A model can be VO-evaluated on MORE THAN ONE test cohort (1105 vs 1806) — different test
        # populations + different GT, so they are NOT one number. The obs/stage2 filenames carry a
        # `_1105_`/`_1806_` tag (from eval_all.sh VO_COHORT_TAG). To surface BOTH as separate board
        # rows, fold the cohort into the row key + display for cohort-tagged VO files ONLY. A file
        # with NO cohort tag (every historical single-cohort run) is unchanged → still one row.
        # (cohort recognition + cohort/arm pseudo-path row keys now live in resolve_vo() /
        #  vo_cohort() at module level — same regex, same anchoring rationale)
        for vj in sorted(VO_RUNS.glob("*.json")):
            name = vj.name.lower()
            if name.endswith(".card.json"):
                continue  # run-card sidecars are routing INPUTS, not result files
            _card = _load_card(vj)  # step-4 card-first routing; None = legacy filename path
            rt = resolve_vo(name, vo_map=VO_FILE_TO_MODEL, vo_exclude=VO_EXCLUDE, card=_card)
            is_two = rt["kind"] == "two_stage"
            is_single = rt["kind"] == "single_stage"
            if not (is_two or is_single):
                continue  # only the two scorable VO families (stage1/agreement are separate)
            if rt["excluded_by"]:
                continue  # probe/variant guard applies regardless of how we resolve the path
            try:
                d = json.loads(vj.read_text())
            except Exception:
                continue
            m = d.get("metrics", {}) or {}
            if not m:
                continue
            # Re-resolve WITH metadata — ALL routing rules live in resolve_vo() (see its
            # docstring): the curated token map (two-stage files record the REASONER in
            # metadata.model, so the filename is the only truth), the single-stage
            # metadata.model fallback, the sft2812 reasoner filter + its three
            # naming-convention exceptions, and the evaluated_samples floors.
            rt = resolve_vo(name, metadata=d.get("metadata") or {},
                            vo_map=VO_FILE_TO_MODEL, vo_exclude=VO_EXCLUDE, card=_card)
            model_path = rt["model_path"]
            if not model_path:
                continue  # no curated token & no fallback → skip (never a wrong join)
            thinking = rt["thinking"]  # two-stage w/o think token = 'off' for JOIN (resolve_vo)
            # ADMISSION GATES — computed by resolve_vo() above: the sft2812-only reasoner
            # rule + its three naming-convention exceptions (cohort bake-off / EXP-B native
            # arms / 397B GT-obs ceiling), and the evaluated_samples floors. Full rationale
            # + the incident history live in resolve_vo()'s body; skip_reason is a distinct
            # sentinel, never a silent drop.
            if not rt["admit"]:
                if rt["skip_reason"].startswith("s1_below_floor:"):
                    _n, _fl = rt["skip_reason"].split(":", 1)[1].split("<", 1)
                    print(f"[vo-s1 SKIP] {vj.name}: evaluated_samples={_n} < floor {_fl} "
                          f"(partial run, not admitted to the board)")
                continue
            # Local aliases for the flags the populate code below still reads (obs-source
            # resolution + per-arm row markers).
            _is_expb_gtobs = rt["is_expb_gtobs"]
            _is_expb_modelobs = rt["is_expb_modelobs"]
            _is_expb_selfloop = rt["is_expb_selfloop"]
            _is_397b_gtobs_ceiling = rt["is_397b_ceiling"]
            _cohort = rt["cohort"]
            _row_path = rt["row_path"]   # cohort + __arm_* pseudo-path suffixes from resolve_vo
            key = rt["row_key"]
            best = ts_best if is_two else ss_best
            tier = rt["tier"]
            if best.get(key, -1) >= tier:
                continue  # a better-or-equal file (within this pipeline) already populated this row
            best[key] = tier
            r = get(_row_path, thinking, display=vj.stem)
            # cohort-tagged rows: show the REAL model path (not the pseudo-key) + a cohort suffix in
            # the display, so the board reads e.g. "…cat k5maj (stage-1) [1806]".
            if _cohort:
                r["model"] = model_path
                if not r.get("_cohort"):
                    r["_cohort"] = _cohort
            if is_two and _is_expb_modelobs and not r.get("_arm"):
                r["_arm"] = "modelobs"
            elif is_two and _is_expb_selfloop and not r.get("_arm"):
                r["_arm"] = "selfloop"
            def vm(k):
                v = m.get(k)
                return round(v * 100, 2) if isinstance(v, (int, float)) else ""
            pfx = "vo_s2_" if is_two else "vo_s1_"
            r[f"{pfx}error_f1"] = vm("error_detection_f1")
            r[f"{pfx}sample_f1"] = vm("sample_error_detection_f1")
            r[f"{pfx}severity_acc"] = vm("overall_severity_accuracy")
            r[f"{pfx}source"] = vj.name
            if not is_two:
                # VO eval completeness for single-stage (mirrors vo_s2_eval_n) — the fix for P1.1:
                # a partial run that clears the floor above is still worth SHOWING as partial, not
                # implying full-N silently.
                _ev1 = (d.get("metadata") or {}).get("evaluated_samples")
                _fl1 = (d.get("metadata") or {}).get("failed_samples")
                if _ev1 is not None:
                    r["vo_s1_eval_n"] = f"{_ev1}/{_fl1}" if _fl1 else str(_ev1)
            # full detail block (all formatted-CSV metrics) for this stage — appended after metadata
            _write_vo_block(r, m, "vo_s2" if is_two else "vo_s1")
            # VO USAGE (two-stage only, 2026-07-07): grounding proxy computed from this file's own
            # per_sample_results. Shuffled control = each VO block scored against the NEXT sample's
            # trace (deterministic shift-by-1); headroom = mean − control is the readable signal
            # (the heuristic matcher fires on unrelated traces often, so the raw mean is inflated).
            if is_two and _vou is not None:
                # Only samples with a NON-EMPTY reasoning trace are measurable: a thinkoff
                # reasoner emits no <think> content, and scoring its samples would render as
                # "100% ungrounded" when the truth is "nothing to measure" (distinct-sentinel
                # rule). Files whose traces are all empty leave the columns BLANK.
                _ps = [s for s in (d.get("per_sample_results") or [])
                       if (s.get("reasoning_content") or "").strip()]
                _scored = [x for x in (_vou.score_sample(s) for s in _ps) if x is not None]
                if _scored:
                    _mu = sum(x["vo_usage_rate"] for x in _scored) / len(_scored)
                    r["vo_s2_usage_mean"] = round(_mu * 100, 2)
                    r["vo_s2_usage_full_pct"] = round(
                        100 * sum(1 for x in _scored if x["vo_usage_rate"] == 1.0) / len(_scored), 2)
                    r["vo_s2_usage_zero_pct"] = round(
                        100 * sum(1 for x in _scored if x["vo_usage_rate"] == 0.0) / len(_scored), 2)
                    # Pairs each sample against a DIFFERENT exercise_id's trace (falls back to a
                    # different session_id) — NOT shift-by-1, since this file is session-grouped
                    # (adjacent samples are usually the same person/exercise, which would make the
                    # control look artificially similar to the real trace). See the function's
                    # docstring in measure_stage2_vo_usage.py for the full rationale.
                    _cm = _vou.shuffled_control_mean(_ps)
                    if _cm is not None:
                        r["vo_s2_usage_ctrl"] = round(_cm * 100, 2)
                        r["vo_s2_usage_headroom"] = round((_mu - _cm) * 100, 2)
            # VO test-set path each eval ran on (1181-rep split) — read per-file, not hardcoded
            _tsd = str((d.get("metadata") or {}).get("test_dataset_dir", "")).strip()
            if _tsd and not r.get("vo_test_set"):
                r["vo_test_set"] = _tsd
            if is_two:
                _md2 = d.get("metadata") or {}
                r["vo_s2_reasoner"] = str(_md2.get("model", "")) or ""
                # VO eval completeness: N evaluated / N failed (surfaces clean 1181/0 vs a partial run).
                _ev = _md2.get("evaluated_samples"); _fl = _md2.get("failed_samples")
                if _ev is not None:
                    r["vo_s2_eval_n"] = f"{_ev}/{_fl}" if _fl else str(_ev)
                # Stage-2 REASONER thinking mode (distinct from the main model's Eval Thinking).
                _st = _md2.get("served_thinking")
                if _st is not None:
                    r["vo_s2_reasoner_thinking"] = "on" if _st else "off"
                # STAGE-2 VOBS SOURCE (2026-07-06) — what fed the stage-2 reasoner's obs block:
                #   - a served MODEL's own stage-1 predictions (real drop-in condition) -> that
                #     model's checkpoint/display name, recovered from precomputed_visual_obs_file's
                #     filename (the obs-gen output naming carries the stage-1 model token) via the
                #     SAME curated map used for VO_FILE_TO_MODEL, so it reads as a real name not a path;
                #   - "GT" -> ground-truth/oracle observations (the EXP-B gtobsbuild arm: GT obs
                #     inlined into the trained prompt itself, single API call, no stage-1 query;
                #     detected by the `_is_expb_gtobs` file-naming convention since metadata alone
                #     can't distinguish "obs baked into prompt" from "no obs at all"). ALSO the
                #     non-EXP-B `_is_397b_gtobs_ceiling` convention (2026-07-09 fix): these files'
                #     own metadata records `pipeline: "single-stage"` / no `precomputed_visual_obs_file`
                #     (the GT obs is inlined into the test-BUILD prompt itself, same as EXP-B's
                #     gtobsbuild — metadata alone can't tell "GT baked into the build" from "no obs
                #     recorded" here either) — without this, both the existing oracle_gtobs_1806
                #     ceiling row AND the new compare_397b_1806_gtobsbuild row fell through to
                #     "unrecorded"/a resolved-obs-source guess and misclassified as
                #     2-stage-FIXED-REASONER-VObs instead of 2-stage-GT-VObs;
                #   - "self" -> the reasoner answered its OWN stage-1 prompt then consumed those
                #     (the selfloop arm) — metadata.stage1_model == metadata.model in this case,
                #     which would otherwise render as a real external model name; call it out
                #     explicitly so the board doesn't misread self-generated obs as sourced obs;
                #   - "unrecorded" -> a two-stage file ALWAYS implies some obs, so this is a
                #     missing-metadata FAILURE, not a real "no obs" condition — distinct sentinel
                #     (never coalesce a metadata gap into a happy-path-shaped value like "none").
                if _is_expb_gtobs or _is_397b_gtobs_ceiling:
                    r["vo_s2_obs_source"] = "GT"
                elif _is_expb_selfloop:
                    r["vo_s2_obs_source"] = "self"
                else:
                    _pvof = str(_md2.get("precomputed_visual_obs_file") or "").strip()
                    if _pvof:
                        _obs_model_path = _vo_model_path(Path(_pvof).name.lower())
                        # Resolve the PATH to its curated DISPLAY name (the board should read a
                        # name like "4B vobs2906 cat k5maj", not a raw /mnt/data/... path).
                        _obs_display = _VO_PATH_TO_DISPLAY.get(_norm_path(_obs_model_path), "") if _obs_model_path else ""
                        r["vo_s2_obs_source"] = _obs_display or _obs_model_path or Path(_pvof).stem
                    else:
                        # No precomputed file -> stage-1 was a live model call; the served
                        # stage-1 model is recorded separately in stage1_model when present.
                        _s1m = str(_md2.get("stage1_model") or "").strip()
                        r["vo_s2_obs_source"] = _s1m or "unrecorded"

        # ---- AGREEMENT vs HUMAN GT (separate family: agreement_*.json). The OLD formatted CSV's
        # "agreement with human annotations" band read error_relevant.vs_gt.a.overall (a = model
        # under test, b = reference): {micro_f1, accuracy, precision, recall}. Verified byte-exact
        # vs the old CSV (27b cat thinkoff: f1 0.2182 / acc 0.6396 / prec 0.3047 / rec 0.17).
        # Same filename→model resolution + VO_EXCLUDE as the scored families (agreement json carries
        # NO metadata.model). Picks the BEST agreement file per (model,thinking) by recency-stable
        # tier (_v2 > v1); never overwrites with a worse one. ----
        agree_best: dict[tuple[str, str], int] = {}
        for aj in sorted(VO_RUNS.glob("agreement_*.json")):
            name = aj.name.lower()
            if name.endswith(".card.json"):
                continue  # run-card sidecars are routing INPUTS, not result files
            rt = resolve_vo(name, vo_map=VO_FILE_TO_MODEL, vo_exclude=VO_EXCLUDE,
                            card=_load_card(aj))
            if rt["excluded_by"]:
                continue
            model_path = rt["model_path"]
            if not model_path:
                continue  # no curated token → skip (distinct sentinel, never a wrong join)
            thinking = rt["thinking"]
            tier = rt["tier"]
            _cohort = rt["cohort"]
            _row_path = rt["row_path"]  # cohort-aware + __arm_* suffixes, same as scored families
            # Arm-aware routing lives in resolve_vo(): DIF-MODEL/OWN-MODEL agreement runs get
            # their own __arm_* row key (caught live 2026-07-06: a modelobs agreement F1 of
            # 52.42 silently merged onto the GT-obs row, indistinguishable from that arm's own
            # 75.90 err-F1), while the `_agree_`-tagged GT-obs-arm agreement is an AXIS of the
            # SAME setup → arm marker only, NO row-key suffix (Sandra correction 2026-07-09).
            _is_expb_modelobs_agree = rt["arm"] == "modelobs"  # -> DIF-MODEL-VObs
            _is_expb_selfloop_agree = rt["arm"] == "selfloop"  # -> OWN-MODEL-VObs
            _is_expb_agree_arm = rt["arm"] == "agree"
            key = rt["row_key"]
            if agree_best.get(key, -1) >= tier:
                continue
            try:
                ad = json.loads(aj.read_text())
                ov = (((ad.get("error_relevant") or {}).get("vs_gt") or {}).get("a") or {}).get("overall") or {}
            except Exception:
                continue
            if not ov:
                continue  # no vs_gt path (e.g. an oracle-only agreement w/o human GT) → skip
            agree_best[key] = tier
            r = get(_row_path, thinking, display=aj.stem)
            if _cohort:
                r["model"] = model_path
                if not r.get("_cohort"):
                    r["_cohort"] = _cohort
            if _is_expb_modelobs_agree and not r.get("_arm"):
                r["_arm"] = "modelobs"
            elif _is_expb_selfloop_agree and not r.get("_arm"):
                r["_arm"] = "selfloop"
            elif _is_expb_agree_arm and not r.get("_arm"):
                r["_arm"] = "agree"
            # reasoner-info columns (normally populated by the two-stage block, which this row
            # never runs through) — an agreement row has no stage-2 call at all, but "which model
            # produced the VObs being scored" is the equivalent question, and it's always THIS
            # checkpoint (self-generated stage-1 obs, live call) — state it explicitly so the row
            # doesn't read as blank/unknown next to the GT-VObs and single-stage rows.
            if _is_expb_agree_arm and not r.get("vo_s2_obs_source"):
                r["vo_s2_reasoner"] = model_path
                r["vo_s2_reasoner_thinking"] = thinking
                r["vo_s2_obs_source"] = f"{Path(model_path).name} (self-generated, agreement-only)"
            def am(k):
                v = ov.get(k)
                return round(v * 100, 2) if isinstance(v, (int, float)) else ""
            r["vo_agree_errf1"] = am("micro_f1")
            r["vo_agree_acc"] = am("accuracy")
            r["vo_agree_prec"] = am("precision")
            r["vo_agree_rec"] = am("recall")
            # matched (rep x error-slot) count actually scored — the P1.1 fix's agreement-side N,
            # so a join over few matched slots (e.g. a stale/short GT file) doesn't read as full-N.
            _n_agree = ov.get("n")
            if isinstance(_n_agree, (int, float)):
                r["vo_agree_eval_n"] = int(_n_agree)
            # "Avg Dist Exercise" (the Variability column) = mean over per-exercise categorical
            # index-distances between the model's answers and the reference (raw-answer agreement,
            # a=model vs b=gt). Lower = closer. RAW (a distance, not a pct). It lives in the agreement
            # JSON's raw-answer block, NOT metrics.*, so we fill the s1 detail block here. The column
            # name says "Exercise" → mean OVER per-exercise distances (not the pooled overall).
            try:
                _pe = (ad.get("per_exercise") or {})
                _dists = [v["categorical"]["mean_index_distance"] for v in _pe.values()
                          if isinstance(v, dict) and isinstance(v.get("categorical"), dict)
                          and isinstance(v["categorical"].get("mean_index_distance"), (int, float))]
                if _dists:
                    r["vo_s1_blk_var_dist"] = round(sum(_dists) / len(_dists), 4)
            except Exception:
                pass

        # ---- ORACLE CEILING row(s) (user 2026-06-23): a DEDICATED reference row = the 397B oracle
        # visual-obs scored vs HUMAN GT = the agreement side=b (b=oracle) constant. It's the upper
        # bound the SFT models distil toward (Agree-F1 0.8608), NOT a deployable model — so it gets its
        # OWN row, distinct from the 397B-plain row (side=a). Driven entirely by master_models.json
        # entries carrying `oracle_ceiling: true` + `vo_agree_side` + `vo_agree_source` (no hardcoded
        # model names here). Keyed on a synthetic path containing the entry's pattern so the allowlist
        # keeps it. ----
        try:
            _cfg = json.loads(MODEL_ALLOWLIST.read_text()) if MODEL_ALLOWLIST.exists() else {}
        except Exception:
            _cfg = {}
        for _e in _cfg.get("models", []):
            if not _e.get("oracle_ceiling"):
                continue
            _side = (_e.get("vo_agree_side") or "b").strip()
            _src = (_e.get("vo_agree_source") or "").strip()
            _aj = VO_RUNS / _src if _src else None
            if not _aj or not _aj.exists():
                print(f"[oracle-ceiling] source missing for {_e.get('pattern')!r}: {_src} — skipped")
                continue
            try:
                _ad = json.loads(_aj.read_text())
                _ov = (((_ad.get("error_relevant") or {}).get("vs_gt") or {}).get(_side) or {}).get("overall") or {}
            except Exception:
                _ov = {}
            if not _ov:
                print(f"[oracle-ceiling] no vs_gt.{_side}.overall in {_src} — skipped")
                continue
            # synthetic key: a non-path string carrying the allowlist pattern so _match_allow keeps it
            # and it never collides with a real served-checkpoint row.
            _key = (_e["pattern"], "off")
            _r = rows.setdefault(_key, {"model": _e["pattern"], "eval_thinking": "off", "owner": ""})
            _r["display"] = _e.get("display", _e["pattern"])
            def _am(k):
                v = _ov.get(k)
                return round(v * 100, 2) if isinstance(v, (int, float)) else ""
            _r["vo_agree_errf1"] = _am("micro_f1")
            _r["vo_agree_acc"] = _am("accuracy")
            _r["vo_agree_prec"] = _am("precision")
            _r["vo_agree_rec"] = _am("recall")
            _r["vo_s1_source"] = f"{_src} (side={_side}, oracle ceiling)"

    # ---- MERGE bare-path aux/bench orphans into their cohort/arm-tagged VO sibling (2026-07-07) ----
    # Aux/benchmark ingestion keys purely on (real served path, thinking) — it has no concept of VO
    # cohort/arm, because aux (multimodal_reduced_testset_1506) and benchmarks are cohort-agnostic by
    # nature. That's fine for every ordinary model. But a cohort/arm-tagged model (EXP-B, vobs2906)
    # whose aux+benchmarks get run in the SAME job as its cohort-specific VO eval creates a SECOND,
    # bare-path row for the same real checkpoint — VO data lands on the `__cohort_*`/`__arm_*` row,
    # aux/bench data lands on a orphaned twin with no cohort tag at all (caught live: EXP-B step562's
    # aux+MMMU/VSI/VMME rendered as their own untagged "27B EXP-B ... step562" row, split from the VO
    # data on "... step562 [1806]"). Fold the aux/bench-only orphan into its cohort/arm sibling ONLY
    # when the merge is unambiguous: the orphan carries no VO data of its own, and exactly one
    # cohort/arm-tagged row for the same (real path, thinking) is missing aux/bench data.
    _VO_FIELD_PREFIXES = ("vo_s1_", "vo_s2_", "vo_agree_")
    def _has_vo_data(rr):
        return any(rr.get(k) not in (None, "") for k in rr if any(k.startswith(p) for p in _VO_FIELD_PREFIXES))
    def _has_aux_or_bench_data(rr):
        return bool(rr.get("aux_acc_weighted_3mod") or rr.get("MMMU_val") or rr.get("Video_MME")
                    or rr.get("VSI_Bench") or rr.get("IF_Bench"))
    _orphan_keys = [k for k in rows if not k[0].endswith(tuple(f"__cohort_{c}" for c in ("1105", "1806")))
                    and "__arm_" not in k[0] and not _has_vo_data(rows[k]) and _has_aux_or_bench_data(rows[k])]
    for _ok in _orphan_keys:
        _orow = rows[_ok]
        _real = _orow.get("model") or _ok[0]
        _think = _ok[1]
        _siblings = [k for k in rows if k != _ok and k[1] == _think and rows[k].get("model") == _real
                     and (k[0].endswith(tuple(f"__cohort_{c}" for c in ("1105", "1806"))) or "__arm_" in k[0])
                     and not _has_aux_or_bench_data(rows[k])]
        # A checkpoint can have MULTIPLE cohort/arm rows (e.g. EXP-B step562 = GT-obs + MODEL-obs
        # arm + SELF-LOOP arm, all sharing this real path) — aux/benchmarks are cohort/arm-AGNOSTIC
        # (they don't vary by which VO obs-source arm was tested), so they belong on the checkpoint's
        # CANONICAL row: the plain cohort-tagged one (`__cohort_*`, no `__arm_*`), not an alternate
        # eval-condition arm. Prefer that when present; only fall back to a bare single match.
        _plain_cohort_siblings = [k for k in _siblings if "__arm_" not in k[0]]
        if len(_plain_cohort_siblings) == 1:
            _siblings = _plain_cohort_siblings
        if len(_siblings) != 1:
            continue  # still ambiguous (0 or >1 candidate) — leave both rows, don't guess
        _sib = rows[_siblings[0]]
        for _k, _v in _orow.items():
            if _k in ("model", "eval_thinking", "owner", "display", "_order", "_group", "_section",
                      "_cohort", "_arm"):
                continue  # identity/curation fields — the sibling's own values win
            if _v not in (None, "") and not _sib.get(_k):
                _sib[_k] = _v
        del rows[_ok]

    # ---- FINALIZE: the two timestamps per row ----
    #  model_created = mtime of the served checkpoint (when the model was created/exported).
    #  last_eval_ts  = newest mtime across the eval artifacts that fed THIS row (aux run dir,
    #                  the benchmark result dirs, the visual-obs JSON) — the last eval performed.
    for (model_path, _think), r in rows.items():
        # For cohort/arm rows, `model_path` (the dict key) is a pseudo-path with a
        # `__cohort_*`/`__arm_*` suffix appended for row-key uniqueness; the REAL served checkpoint
        # path lives in `r["model"]` (set explicitly wherever a cohort/arm tag is applied above).
        # Using the raw key here made model_created mtime-lookups miss (no such file) -> blank on
        # every cohort/arm row (2026-07-06 audit fix).
        _real_model_path = r.get("model") or model_path
        r["is_baseline"] = "yes" if _is_baseline(_real_model_path, r.get("display", "")) else "no"
        # model_created = ckpt export mtime. Meaningless for a BASELINE (bare upstream Qwen3.5 —
        # the dir mtime is just when the shared files synced to disk, not a real creation date) ->
        # leave blank for baselines.
        r["model_created"] = "" if r["is_baseline"] == "yes" else _model_created(_real_model_path)
        # benchmark result dirs are keyed by the BENCHMARK display (`_bench_display`), NOT the row's
        # `display` (which is the aux `<base>:<run_id>` for any model with an aux row, since aux runs
        # first). Using `display` here made BENCH_RESULTS/<bench>/<aux-display> never exist → benchmark
        # mtime never counted toward last_eval_ts. Fall back to `display` only when no bench ran.
        bench_disp = r.get("_bench_display") or r.get("display", "")
        ev_paths = [r.get("aux_run_dir", "")]
        # ifbench added (2026-07-06 audit fix): this loop omitted it, so an IFBench-only re-run's
        # mtime never counted toward last_eval_ts (the board's "Last Eval" column could understate
        # how recently a model was actually touched).
        for bench in ("mmmu_val", "video_mme", "vsibench", "ifbench"):
            ev_paths.append(str(BENCH_RESULTS / bench / bench_disp))
        for vo in (r.get("vo_s1_source", ""), r.get("vo_s2_source", "")):
            if vo:
                ev_paths.append(str(VO_RUNS / vo))
        r["last_eval_ts"] = _fmt_ts(_newest_mtime(*ev_paths))

    # ---- DEDUP stale baseline rows ----
    # A baseline scored under an untagged aux run-id lands as thinking='unknown'. When a properly
    # tagged (on/off) row exists for the SAME canonical model, the unknown row is a stale duplicate
    # (older aux-only run) — drop it so each baseline shows one row per real thinking mode.
    tagged_baselines = {mp for (mp, th), r in rows.items()
                        if r.get("is_baseline") == "yes" and th in ("on", "off")}
    for key in [k for k in rows if k[1] == "unknown" and rows[k].get("is_baseline") == "yes"
                and k[0] in tagged_baselines]:
        del rows[key]

    # ---- FAMILY (2026-07-09, see ~/.claude/EVAL_MAP.md) ----
    # Classify each row's taxonomy family from which metric blocks are ACTUALLY populated (never
    # from the display string, which can be stale — that ambiguity is exactly what caused the
    # 2026-07-09 flipfix step840 incident this column exists to prevent). Computed here, last, so
    # it can just check `r.get(col)` against the same fields every other column already wrote.
    for r in rows.values():
        r["family"] = _classify_family(r)

    # ---- VISIBILITY for the two silent-acceptance paths the critique flagged ----
    # (1) aux runs admitted to the 1506 axis ONLY by the date fallback (no explicit 1506 token) —
    #     allow-by-absence, so surface them; a new experiment on yet-another testset shows up here.
    if datefallback_runs:
        print(f"[aux-testset] {len(datefallback_runs)} run(s) classed 1506 by DATE FALLBACK "
              f"(no explicit 1506 token; stamped >= {_V2_TS_BOUNDARY}):")
        for rid in datefallback_runs:
            print(f"             · {rid}")
    # (2) ORPHAN rows: an aux fallback that couldn't find RUN_METADATA.model synthesizes a
    #     non-path key '<base>/<run_id>' that _norm_path can't resolve, so it never merges with the
    #     real-path bench/VO rows for that model — a fragmentary orphan. Count + name them.
    #     Exclude INTENTIONAL synthetic keys (oracle-ceiling rows key on the allowlist `pattern`
    #     string by design, per the oracle_ceiling block above — not an aux-join failure). Reload
    #     the allowlist independently here (not the `_cfg` local from the VO_RUNS block, which
    #     doesn't exist when VO_RUNS is absent — this check must not depend on that branch running).
    try:
        _oc_cfg = json.loads(MODEL_ALLOWLIST.read_text()) if MODEL_ALLOWLIST.exists() else {}
    except Exception:
        _oc_cfg = {}
    _oracle_ceiling_keys = {(_e["pattern"], "off") for _e in _oc_cfg.get("models", []) if _e.get("oracle_ceiling")}
    orphans = [(mp, th) for (mp, th) in rows
               if mp and not mp.startswith("/") and (mp, th) not in _oracle_ceiling_keys]
    if orphans:
        print(f"[orphan-rows] {len(orphans)} row(s) keyed on a synthesized non-path string "
              f"(no RUN_METADATA.model → won't merge with bench/VO for that model):")
        for mp, th in orphans:
            print(f"             · {mp} [{th}]")

    return rows


def _resolve_vo_from_card(name: str, card: dict, metadata: dict | None, rt: dict) -> dict:
    """Card-first routing (see resolve_vo docstring). Identity = the card's fields; the
    filename contributes only the _v2/_cat tier. A malformed card degrades LOUDLY: any
    missing required field sets admit=False with a distinct card_invalid sentinel —
    never a silent fall-through to filename guessing (feedback_no_silent_fail)."""
    import math
    rt["carded"] = True
    axis = str(card.get("axis") or "").strip().lower()
    kind = {"singlestage": "single_stage", "stage2": "two_stage",
            "agreement": "agreement"}.get(axis)
    ckpt = str(card.get("checkpoint_path") or "").strip()
    thinking = str(card.get("thinking") or "").strip().lower()
    if kind is None or not ckpt or thinking not in ("on", "off"):
        rt.update(admit=False,
                  skip_reason=f"card_invalid:axis={axis!r},ckpt={'set' if ckpt else 'MISSING'},"
                              f"thinking={thinking!r}")
        return rt
    rt["kind"] = kind
    rt["model_path"] = ckpt
    rt["thinking"] = thinking
    rt["cohort"] = str(card.get("cohort") or "").strip()
    arm = str(card.get("arm") or "").strip().lower()
    obs_source = str(card.get("obs_source") or "").strip()
    rt["is_expb_gtobs"] = arm == "gtobsbuild" or obs_source.upper() == "GT"
    rt["is_expb_modelobs"] = arm == "modelobs"
    rt["is_expb_selfloop"] = arm == "selfloop" or obs_source.lower() == "self"
    rt["arm"] = arm if arm in ("modelobs", "selfloop", "agree") else ""
    if kind == "agreement":
        rt["tier"] = 1 if name.endswith("_v2.json") else 0
    else:
        rt["tier"] = (10 if name.endswith("_v2.json") else 0) + (1 if "_cat" in name else 0)
    row_path = f"{ckpt}__cohort_{rt['cohort']}" if rt["cohort"] else ckpt
    if rt["arm"] in ("modelobs", "selfloop"):
        row_path += f"__arm_{rt['arm']}"
    rt["row_path"] = row_path
    rt["row_key"] = (_norm_path(row_path), thinking)
    _expected = card.get("expected_n")
    rt["floor"] = (math.floor(_expected * 0.99) if isinstance(_expected, (int, float)) and _expected
                   else (2237 if rt["cohort"] == "1806" else 1169 if rt["cohort"] == "1105" else 1170))
    if metadata is not None and kind in ("two_stage", "single_stage"):
        _eN = metadata.get("evaluated_samples") or 0
        if _eN < rt["floor"]:
            pfx = "s1_" if kind == "single_stage" else ""
            rt.update(admit=False, skip_reason=f"{pfx}below_floor:{_eN}<{rt['floor']}")
    if kind == "two_stage":
        if rt["is_expb_gtobs"]:
            rt["obs_source_hint"] = "GT"
            rt["family_hint"] = "2-stage-GT-VObs"
        elif rt["is_expb_selfloop"]:
            rt["obs_source_hint"] = "self"
            rt["family_hint"] = "2-stage-OWN-MODEL-VObs"
        elif rt["is_expb_modelobs"]:
            rt["family_hint"] = "2-stage-DIF-MODEL-VObs"
        else:
            rt["family_hint"] = "2-stage-FIXED-REASONER-VObs"
    elif kind == "single_stage":
        rt["family_hint"] = "Single-stage"
    else:
        rt["family_hint"] = "VObs-agreement"
    return rt


def _load_card(json_path: Path) -> dict | None:
    """Read `<file>.card.json` if present/valid; None otherwise (legacy file)."""
    cp = Path(str(json_path) + ".card.json")
    if not cp.exists():
        return None
    try:
        c = json.loads(cp.read_text())
        return c if isinstance(c, dict) else None
    except Exception as e:
        print(f"[card WARN] {cp.name}: unreadable ({e}) — falling back to filename routing")
        return None


def _route_report(paths) -> int:
    """ROUTING DRY-RUN (`--route`): print, for each VO result filename (existing file OR a
    planned name that doesn't exist yet), exactly where the compile pass will put it — row
    key, cohort/arm, family, floor, admission, and which master_models.json entry claims
    its display. Launches and writes NOTHING. Exit 1 if ANY input would be invisible on the
    board (no curated token / no allowlist entry / not a routed VO file) so this can gate a
    preflight. Uses the SAME resolve_vo() as the compile pass — simulation cannot diverge."""
    vo_map, vo_exclude = _vo_config_cached()
    allow = _load_allowlist() or []
    bad = 0
    for p in paths:
        fp = Path(p)
        md = None
        note = "planned name (no file on disk) — metadata gates (reasoner/floor) not checked"
        if fp.exists():
            try:
                md = (json.loads(fp.read_text()).get("metadata") or {})
                note = ""
            except Exception as e:
                note = f"file exists but unreadable ({e}) — metadata gates not checked"
        rt = resolve_vo(fp.name, metadata=md, vo_map=vo_map, vo_exclude=vo_exclude,
                        card=_load_card(fp))
        print(f"\n=== {fp.name}")
        if rt["carded"]:
            print("  card     : ✓ routed from <file>.card.json (no vo_tokens needed)")
        if note:
            print(f"  note     : {note}")
        if rt["kind"] is None:
            print("  kind     : ⚠ NOT a routed VO file (no stage2_/agreement_ prefix or "
                  "singlestage marker) — the compiler will never read it")
            bad += 1
            continue
        print(f"  kind     : {rt['kind']}   family: {rt['family_hint']}")
        print(f"  cohort   : {rt['cohort'] or '(none — joins the bare-path row)'}"
              f"   arm: {rt['arm'] or '(none)'}   thinking: {rt['thinking']}")
        if rt["excluded_by"]:
            print(f"  admit    : ✗ excluded by vo_exclude token {rt['excluded_by']!r}")
            continue  # deliberate exclusion — not counted as a routing failure
        if not rt["model_path"]:
            print("  model    : ⚠ NONE — no vo_tokens match ⇒ INVISIBLE ON THE BOARD."
                  " Add a master_models.json entry whose vo_tokens is a literal substring"
                  " of this filename (mind dash vs underscore).")
            bad += 1
            continue
        print(f"  model    : {rt['model_path']}")
        print(f"  row key  : {rt['row_key']}")
        print(f"  floor    : evaluated_samples ≥ {rt['floor']}"
              + (f"   admit: {'✓' if rt['admit'] else '✗ ' + rt['skip_reason']}" if md is not None else ""))
        entry = _match_allow({"model": rt["model_path"] if rt["cohort"]
                              else _norm_path(rt["model_path"]),
                              "display": fp.name.rsplit(".json", 1)[0]}, allow)
        if entry is None:
            print("  display  : ⚠ NO master_models.json `pattern` matches ⇒ row DROPPED by the"
                  " allowlist (invisible). Add/extend the entry BEFORE launching.")
            bad += 1
        else:
            print(f"  display  : {entry.get('display') or '(raw)'}   "
                  f"(pattern {entry.get('pattern')!r})")
    print(f"\n[route] {len(paths)} file(s) checked, {bad} would be invisible/unrouted."
          + ("" if not bad else "  FIX BEFORE LAUNCH."))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-copy", action="store_true", help="skip copying per-run summaries into eval_master/runs/")
    ap.add_argument("--route", nargs="+", metavar="FILE",
                    help="routing dry-run: report the board row each VO result filename "
                         "(existing or PLANNED) will land on, then exit — writes nothing. "
                         "The routing-integrity preflight (stabilization plan 2026-07-10).")
    args = ap.parse_args()

    if args.route:
        return _route_report(args.route)

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()

    # New-era curation: keep only allowlisted models in the LIVE master files (the full
    # historical dump is frozen under results/master/v1/). If the allowlist file is absent,
    # keep ALL rows (legacy behavior) so nothing silently vanishes.
    allow = _load_allowlist()
    items = list(rows.items())
    if allow is not None:
        kept = []
        for k, r in items:
            entry = _match_allow(r, allow)
            if entry is None:
                continue
            # keep_thinking: drop a model's row whose thinking mode isn't the one we want to show
            # (e.g. reasoning-ep3 keep_thinking='on' → drop its degenerate thinkoff contrast row).
            if entry.get("keep_thinking") and r.get("eval_thinking") != entry["keep_thinking"]:
                continue
            if entry["display"]:               # curated clean display name overrides the raw one
                r["display"] = entry["display"]
            # COHORT/ARM DISPLAY SUFFIX (2026-07-06 audit fix, P1.3): the curated display override
            # above previously clobbered the cohort tag stashed on `r["_cohort"]` in `_rows()`
            # (comment there promised "[1806]" but nothing ever rendered it) — 4 models' 1105-cohort
            # and 1806-cohort rows shared one identical display string with materially different
            # numbers, distinguishable only via the far-right VO Test Set column. Apply it HERE,
            # after any curated-display override, so it can never be silently dropped again.
            if r.get("_cohort") and f"[{r['_cohort']}]" not in r["display"]:
                r["display"] = f"{r['display']} [{r['_cohort']}]"
            # Only auto-append the arm label if the curated display doesn't already spell it out
            # (some allowlist entries, like the EXP-B modelobs/selfloop rows, curate their own
            # "(...arm)" text — this is the fallback for any FUTURE arm row without a curated label).
            if r.get("_arm") == "modelobs" and "MODEL-obs arm" not in r["display"]:
                r["display"] = f"{r['display']} (MODEL-obs arm)"
            elif r.get("_arm") == "selfloop" and "SELF-LOOP arm" not in r["display"]:
                r["display"] = f"{r['display']} (SELF-LOOP arm)"
            if entry["train_reasoning"]:        # curated train_reasoning is AUTHORITATIVE — it OVERRIDES
                # the eval_matrix value. The matrix's train_reasoning is auto-derived per aux-run and is
                # WRONG for some (e.g. pmartins sft2812/grpo492 trained on diverse_reasoning+mix_reas
                # were stamped 'No' by the aux export → board showed 'No' despite the curated 'yes').
                # The allowlist is the hand-verified source of truth, so it wins (2026-06-22).
                r["train_reasoning"] = entry["train_reasoning"]
            if entry.get("train_sample_count"):   # curated — eval_matrix train_sample_count is unwired for V2
                r["train_sample_count"] = entry["train_sample_count"]
            r["_order"] = entry["order"]        # allowlist position (board ordering)
            r["_group"] = entry["group"]        # group bucket (blank row inserted between groups)
            r["_section"] = entry["section"]    # section banner label (see _load_allowlist)
            kept.append((k, r))
        dropped = len(items) - len(kept)
        print(f"[allowlist] {MODEL_ALLOWLIST.name}: {len(allow)} patterns -> kept {len(kept)}/{len(items)} rows ({dropped} off-board)")
        items = kept
    else:
        print(f"[allowlist] {MODEL_ALLOWLIST.name} absent -> keeping ALL {len(items)} rows (legacy)")

    # combined master (all curated families), baselines pinned to top
    n = _write_csv(MASTER_CSV, items, FIELDS)
    print(f"[master] wrote {n} rows -> {MASTER_CSV}")

    # per-base-model splits — only the active-era families (4b, 27b); baselines pinned to top
    from collections import defaultdict
    by_base = defaultdict(list)
    for key, row in items:
        by_base[_base_model(row.get("model", ""), row.get("display", ""))].append((key, row))
    for fam in ERA_FAMILIES:
        fam_csv = MASTER_DIR / f"eval_master_{fam}.csv"
        nf = _write_csv(fam_csv, by_base.get(fam, []), FIELDS)
        nbase = sum(_is_baseline(r.get("model", ""), r.get("display", "")) for _k, r in by_base.get(fam, []))
        print(f"[master:{fam}] wrote {nf} rows ({nbase} baseline) -> {fam_csv}")
    other = sorted(set(by_base) - set(ERA_FAMILIES))
    if other:
        n_other = sum(len(by_base[f]) for f in other)
        print(f"[master] note: {n_other} curated rows in non-era families {other} are in the combined CSV but have no split file")

    # copy per-run aux aggregate JSON + SUMMARY (small, human-readable) into eval_master/runs/
    if not args.no_copy and AUX_EVALS.is_dir():
        COPY_DIR.mkdir(parents=True, exist_ok=True)
        n = 0
        for agg in AUX_EVALS.glob("*/multimodal/*/*/*/*/results/multimodal_*.json"):
            run_dir = agg.parent.parent  # .../<run_id>/<ts>/
            label = f"{run_dir.parent.name}__{run_dir.name}"  # <run_id>__<ts>
            dest = COPY_DIR / label
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(agg, dest / agg.name)
            for s in run_dir.glob("SUMMARY_*"):
                shutil.copy2(s, dest / s.name)
            n += 1
        print(f"[copy] synced {n} aux run summaries -> {COPY_DIR}")

    print("\nNOTE: rows are JOINED on the served checkpoint PATH (canonical key). A model joins")
    print("across stages only where it was actually evaluated on each; single-stage models stay")
    print("single-stage rows. 'thinking=unknown' = a source file whose name lacked _thinkon/off.")
    print("READ-ONLY union/join; the per-stage CSVs remain the source of truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
