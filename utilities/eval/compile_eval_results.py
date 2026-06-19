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

# one row per (model_key, thinking); model_key = the served model basename / display name.
# Column order = logical reading order: IDENTITY -> WHEN -> HEADLINE SCORES (benchmarks +
# aux + visual-obs) -> AUX DETAIL -> TRAINING PROVENANCE -> SOURCE PROVENANCE (bookkeeping).
FIELDS = [
    # --- identity: who/what this row is (train_reasoning sits right before eval_thinking) ---
    "display", "model", "model_created", "owner", "is_baseline", "train_reasoning", "eval_thinking",
    # --- when: most-recent eval (model_created is up with identity) ---
    "last_eval_ts",
    # --- headline scores: the numbers you scan first ---
    #   general benchmarks (+ per-cell scoring method: parsable / judged / raw)
    "MMMU_val", "Video_MME", "VSI_Bench", "bench_method",
    #   visual-obs headline (two-stage severity/detection)
    "vo_error_f1", "vo_sample_f1", "vo_severity_acc",
    #   visual-obs AGREEMENT vs HUMAN GT (single-stage obs; error_relevant.vs_gt.a.overall) —
    #   the comparable no-reasoner signal the old formatted CSV showed as its own band
    "vo_agree_errf1", "vo_agree_acc", "vo_agree_prec", "vo_agree_rec",
    #   aux 3-modality headline (after visual-obs)
    "aux_acc_weighted_3mod",
    # --- aux per-modality / per-task detail. Video splits by source: 3D MCQA (mcqa_video_3d_2705,
    # the harder spatial-reasoning set) vs non-3D (mcqa_video_1505). Combined = aux_video_acc. ---
    "aux_video_acc", "aux_video_3d", "aux_video_non3d", "aux_text_acc", "aux_image_composite",
    "aux_image_dense_oks", "aux_image_task4_acc",
    # --- training provenance (from eval_matrix; train_reasoning moved up to identity) ---
    "train_group_id", "train_sample_count", "best_step",
    # --- source provenance / bookkeeping (last) ---
    "aux_run_ts", "aux_run_id", "aux_run_dir", "aux_source", "bench_source", "vo_source",
]

# Human-readable header labels for the CSV (so a paste into Excel reads cleanly). Internal field
# KEYS stay snake_case everywhere in code; only the written header row is prettified. Any field
# not listed falls back to its key with '_'->' ' and title-cased.
HEADER_LABELS = {
    "display": "Display", "model": "Model Path", "model_created": "Model Created",
    "owner": "Owner", "is_baseline": "Baseline?", "train_reasoning": "Trained w/ Reasoning",
    "eval_thinking": "Eval Thinking", "last_eval_ts": "Last Eval",
    "MMMU_val": "MMMU-val", "Video_MME": "Video-MME", "VSI_Bench": "VSI-Bench",
    "bench_method": "Benchmark Scoring",
    "vo_error_f1": "VO Error-F1", "vo_sample_f1": "VO Sample-F1", "vo_severity_acc": "VO Severity Acc",
    "vo_agree_errf1": "VO Agree-F1 (vs GT)", "vo_agree_acc": "VO Agree-Acc (vs GT)",
    "vo_agree_prec": "VO Agree-Prec (vs GT)", "vo_agree_rec": "VO Agree-Rec (vs GT)",
    "aux_acc_weighted_3mod": "Aux 3-Mod Weighted",
    "aux_video_acc": "Aux Video (all)", "aux_video_3d": "Aux Video 3D", "aux_video_non3d": "Aux Video non-3D",
    "aux_text_acc": "Aux Text", "aux_image_composite": "Aux Image",
    "aux_image_dense_oks": "Aux Image Dense OKS", "aux_image_task4_acc": "Aux Image Task4",
    "train_group_id": "Train Group", "train_sample_count": "Train Samples", "best_step": "Best Step",
    "aux_run_ts": "Aux Run TS", "aux_run_id": "Aux Run ID", "aux_run_dir": "Aux Run Dir",
    "aux_source": "Aux Source", "bench_source": "Benchmark Source", "vo_source": "VO Source",
}


def _header(field: str) -> str:
    return HEADER_LABELS.get(field, field.replace("_", " ").title())


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
                "group": (e.get("group") or "").strip(),
                "order": idx,  # board rows appear in allowlist order; blank row between groups
            })
    return out


def _match_allow(row: dict, allow):
    """Return the FIRST matching allowlist entry dict for a row, or None.
    Match is a substring of the row's model path OR display."""
    hay = f"{row.get('model','')} {row.get('display','')}".lower()
    for entry in allow:
        if entry["pattern"] in hay:
            return entry
    return None


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
    if timestamp and timestamp[:10] >= _V2_TS_BOUNDARY:  # new-pipeline run (eval_all -> 1506)
        return "1506"
    return "unknown"


def _write_csv(path: Path, row_items, fields):
    """Write rows. If rows carry an allowlist `_order` (curated board), emit in THAT order and
    insert a BLANK row between groups (`_group` change) — the user-defined layout. Otherwise
    (legacy / no allowlist) pin baselines to the top, then alphabetical by model."""
    have_order = any("_order" in r for _k, r in row_items)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        csv.writer(f).writerow([_header(k) for k in fields])  # pretty (human-readable) header row
        if have_order:
            # within an allowlist position, keep a stable secondary sort (thinkoff before thinkon)
            _th = {"off": 0, "on": 1, "unknown": 2}
            ordered = sorted(row_items, key=lambda kv: (kv[1].get("_order", 1e9),
                                                        _th.get(kv[1].get("eval_thinking", ""), 3)))
            n, prev_group = 0, None
            for _key, row in ordered:
                g = row.get("_group", "")
                if prev_group is not None and g != prev_group:
                    w.writerow({k: "" for k in fields})  # blank separator row between groups
                w.writerow({k: row.get(k, "") for k in fields})
                prev_group = g
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


def _parsable_bench_acc(disp: str, bench: str):
    """Recompute a benchmark's accuracy over PARSABLE answers only, from the per-sample
    *_result.xlsx VLMEvalKit writes (cols: hit, prediction). Excludes 'Failed to obtain answer'
    non-responses from BOTH numerator and denominator. Returns (pct, n_used, n_dropped) or None
    if no result file / no pandas. The raw summary.csv value remains the fallback."""
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
    bad = pred.eq("") | pred.str.contains("failed to obtain answer", na=False) | pred.str.contains("api error", na=False)
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
    for bench in ("mmmu_val", "video_mme", "vsibench",
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


def _rows():
    """Return {(model_path, thinking): {field: value}} JOINED on the served checkpoint path."""
    rows: dict[tuple[str, str], dict] = {}

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
                if _aux_testset(rec.get("run_id", ""), rec.get("eval_family", ""), rec.get("tag", ""), rec.get("timestamp", "")) != "1506":
                    continue
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
                r["aux_image_task4_acc"] = _f(rec.get("acc_task4a")) or _f(rec.get("acc_task4b"))
                r["train_reasoning"] = rec.get("train_reasoning", "")
                r["train_group_id"] = rec.get("train_group_id", "")
                r["train_sample_count"] = rec.get("train_sample_count", "")
                r["best_step"] = rec.get("best_step", "")
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
            if _aux_testset(run_id, str(d.get("eval_family", "")), str(d.get("tag", "")), str(d.get("created_at", ""))) != "1506":
                continue
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
                def num(c):
                    v = (rec.get(c) or "").strip()
                    try:
                        return round(float(v) * 100, 2) if v and float(v) <= 1.0 else (round(float(v), 2) if v else "")
                    except ValueError:
                        return ""
                mmmu, vmme, vsi = num("MMMU-val"), num("Video-MME"), num("VSI-Bench")
                # which summary file is this? raw=non-responses counted wrong; judged=LLM-judged
                method = "judged" if "judge" in bench_csv.name.lower() else "raw"
                meth = dict(_kv.split("=", 1) for _kv in (r.get("bench_method") or "").split(";") if "=" in _kv)
                # only overwrite a column when this file actually has a value for it
                if mmmu != "": r["MMMU_val"] = mmmu; meth["MMMU"] = method
                if vmme != "": r["Video_MME"] = vmme; meth["VMME"] = method
                if vsi != "": r["VSI_Bench"] = vsi; meth["VSI"] = method
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
                    pa = _parsable_bench_acc(disp, bench)
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
    # Curated map: (anchor-token, served path). Anchors are SPECIFIC to the carried champion so
    # they don't catch sibling probes (OBSGUIDE/TEXTONLY/plus_mix12k/_A_/_B_ sweeps are NOT the
    # carried model). VO_EXCLUDE rejects a file even if a token matches (probe/variant guard).
    VO_FILE_TO_MODEL = [  # ordered; first matching token wins
        ("oracle_obs_cat_union5_step339",  "/mnt/data/sgsilva/models/qwen35-27b-oracle-obs-cat-union5-step339"),
        ("union_oracleobs_llmfms_ep3",     "/mnt/data/sgsilva/models/qwen35-27b-oracle-obs-cat-plus-llm-fms-step1785"),
        ("oracle_obs_merged_1805_step2558","/mnt/data/sgsilva/models/qwen35-27b-oracle-obs-merged-1805-step2558"),
        ("oracle_obs_cat_step357",         "/mnt/data/sgsilva/models/qwen35-4b-oracle-obs-cat-1105-step357"),
        ("reasoning_oracleobs_cat_ep3",    "/mnt/data/sgsilva/models/qwen35-27b-oracle-obs-cat-reasoning-step330"),
        ("oracle_397b_categorical",        "/mnt/data/shared/models/Qwen3.5-397B-A17B"),
        # bare baselines (two-stage cat preferred; canonical path collapses via _BASELINE_ALIASES)
        ("baseline_qwen35_27b_cat",        "/mnt/data/shared/models/Qwen3.5-27B"),
        ("baseline_qwen35_27b",            "/mnt/data/shared/models/Qwen3.5-27B"),
        ("baseline_qwen35_4b",             "/mnt/data/shared/models/Qwen3.5-4B"),
        # historical AGREEMENT files name the bare baseline with a DOTTED id (agreement_qwen3.5-27b_
        # cat_…), not the baseline_qwen35_NB token the scored files use. Add the dotted variants
        # LAST (broad substrings; first-match precedence means specific SFT tokens above win first).
        ("qwen3.5-27b",                    "/mnt/data/shared/models/Qwen3.5-27B"),
        ("qwen3.5-4b",                     "/mnt/data/shared/models/Qwen3.5-4B"),
    ]
    # files whose token matches but are a DIFFERENT model / non-deployable probe — never join.
    VO_EXCLUDE = ("obsguide", "textonly", "plus_mix12k", "_a_ep3", "_b_ep3", "_c_ep3", "_d_ep3",
                  "baseline_ep3", "reasoner-self", "union5_decode")
    def _vo_model_path(name: str) -> str:
        """Resolve a VO JSON filename -> served checkpoint path via the curated map. '' if no
        match OR an excluded probe (distinct sentinel — the row is then skipped, not joined to a
        wrong key). Add a carried model's anchor token here when its VO run lands."""
        if any(x in name for x in VO_EXCLUDE):
            return ""
        for tok, path in VO_FILE_TO_MODEL:
            if tok in name:
                return path
        return ""
    def _vo_tier(name: str) -> int:
        """Higher = preferred. stage2 beats singlestage; _v2 rescored beats v1 within a tier."""
        base = 2 if name.startswith("stage2_") else (1 if "singlestage" in name else 0)
        return base * 2 + (1 if name.endswith("_v2.json") else 0)
    if VO_RUNS.is_dir():
        vo_best: dict[tuple[str, str], int] = {}  # (model,thinking) -> winning tier so far
        for vj in sorted(VO_RUNS.glob("*.json")):
            name = vj.name.lower()
            if "singlestage" not in name and not name.startswith("stage2_"):
                continue  # only the two scorable VO families (stage1/agreement are separate)
            tier = _vo_tier(name)
            if tier == 0:
                continue
            if any(x in name for x in VO_EXCLUDE):
                continue  # probe/variant guard applies regardless of how we resolve the path
            try:
                d = json.loads(vj.read_text())
            except Exception:
                continue
            m = d.get("metrics", {}) or {}
            if not m:
                continue
            # Resolve the VO model path: prefer the curated historical map (two-stage files record
            # the REASONER in metadata.model, so the filename is the only truth). For a NEW-PIPELINE
            # run not in the map, fall back to metadata.model — eval_all.sh visualobs is single-stage
            # and DOES record the real served path there. Skip if neither resolves (no garbage key).
            model_path = _vo_model_path(name)
            # Fallback to metadata.model ONLY for a genuinely NEW-PIPELINE single-stage file —
            # i.e. one that carries no historical VO model token at all (so we don't resurrect V1
            # single-stage noise for a model the curated map already scores via its stage2 file).
            historical = any(t in name for t in (
                "oracleobs", "oracle_obs", "llmfms", "union5", "397b", "baseline_qwen35", "mix12k"))
            if not model_path and "singlestage" in name and not historical:
                model_path = str((d.get("metadata") or {}).get("model", "")) or ""
            if not model_path:
                continue
            thinking = "on" if "thinkon" in name else ("off" if "thinkoff" in name else "unknown")
            key = (_norm_path(model_path), thinking)
            if vo_best.get(key, -1) >= tier:
                continue  # a better-or-equal VO file already populated this row
            vo_best[key] = tier
            r = get(model_path, thinking, display=vj.stem)
            def vm(k):
                v = m.get(k)
                return round(v * 100, 2) if isinstance(v, (int, float)) else ""
            r["vo_error_f1"] = vm("error_detection_f1")
            r["vo_sample_f1"] = vm("sample_error_detection_f1")
            r["vo_severity_acc"] = vm("overall_severity_accuracy")
            r["vo_source"] = vj.name

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
            if any(x in name for x in VO_EXCLUDE):
                continue
            model_path = _vo_model_path(name)
            if not model_path:
                continue  # no curated token → skip (distinct sentinel, never a wrong join)
            thinking = "on" if "thinkon" in name else ("off" if "thinkoff" in name else "unknown")
            tier = 1 if name.endswith("_v2.json") else 0
            key = (_norm_path(model_path), thinking)
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
            r = get(model_path, thinking, display=aj.stem)
            def am(k):
                v = ov.get(k)
                return round(v * 100, 2) if isinstance(v, (int, float)) else ""
            r["vo_agree_errf1"] = am("micro_f1")
            r["vo_agree_acc"] = am("accuracy")
            r["vo_agree_prec"] = am("precision")
            r["vo_agree_rec"] = am("recall")

    # ---- FINALIZE: the two timestamps per row ----
    #  model_created = mtime of the served checkpoint (when the model was created/exported).
    #  last_eval_ts  = newest mtime across the eval artifacts that fed THIS row (aux run dir,
    #                  the benchmark result dirs, the visual-obs JSON) — the last eval performed.
    for (model_path, _think), r in rows.items():
        r["is_baseline"] = "yes" if _is_baseline(model_path, r.get("display", "")) else "no"
        # model_created = ckpt export mtime. Meaningless for a BASELINE (bare upstream Qwen3.5 —
        # the dir mtime is just when the shared files synced to disk, not a real creation date) ->
        # leave blank for baselines.
        r["model_created"] = "" if r["is_baseline"] == "yes" else _model_created(model_path)
        disp = r.get("display", "")
        ev_paths = [r.get("aux_run_dir", "")]
        for bench in ("mmmu_val", "video_mme", "vsibench"):
            ev_paths.append(str(BENCH_RESULTS / bench / disp))
        vo = r.get("vo_source", "")
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

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-copy", action="store_true", help="skip copying per-run summaries into eval_master/runs/")
    args = ap.parse_args()

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
            if entry["display"]:               # curated clean display name overrides the raw one
                r["display"] = entry["display"]
            if entry["train_reasoning"] and not r.get("train_reasoning"):  # curated fills the gap
                r["train_reasoning"] = entry["train_reasoning"]
            r["_order"] = entry["order"]        # allowlist position (board ordering)
            r["_group"] = entry["group"]        # group bucket (blank row inserted between groups)
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
