#!/usr/bin/env python3
"""
Video SFT Dataset Monitor.

Browse generated MCQA samples and explore exercise types from the Thrive VLM Database.

Usage:
    cd /home/sgsilva/utilities/apps/video_sft
    source .venv/bin/activate
    python app.py --port 7862
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import string
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set temp dir before importing gradio
os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/.gradio_temp")

# Relocated 2026-06-30 out of the archived video-sft-vlm repo into
# ~/utilities/apps/video_sft/. The repo's stale scripts/ + utils/ were NOT copied —
# they were duplicates of newer canonical modules already in vlm-post-training. We
# import those canonical copies directly so there is ONE source of truth:
#   metric_calculator      -> video_mcqa/evaluation/   (May 14, newest)
#   build_pkg, pkg_visualizer -> shared/pkg/           (Jun 29, newest)
#   utils.{data_loader,geometry,keypoint_adapter} -> video_mcqa/generation/utils/
_VPT = Path(os.environ.get("VLM_POST_TRAINING_ROOT", "/home/sgsilva/vlm-post-training"))
_VMCQA = _VPT / "aux_tasks" / "video_tasks" / "video_mcqa"
for _p in (
    str(Path(__file__).resolve().parent),       # this package (for data/)
    str(_VMCQA / "generation"),                  # exposes `utils` package
    str(_VMCQA / "evaluation"),                  # metric_calculator
    str(_VPT / "aux_tasks" / "shared" / "pkg"),  # build_pkg, pkg_visualizer
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# In-package data home (annotations write-back + pkg_graph fallback). pkg_graph.json
# now lives in vlm-post-training/aux_tasks/shared/data/ (its canonical home).
DATA_DIR = SCRIPT_DIR / "data"
PKG_GRAPH_PATH = os.environ.get(
    "PKG_GRAPH_JSON",
    "/home/sgsilva/vlm-post-training/aux_tasks/shared/data/pkg_graph.json",
)

CONFIG = {
    "data_dir": os.environ.get("DATA_DIR", "/mnt/data/shared/vlm/data/10k/all"),
    "default_jsonl": os.environ.get("DEFAULT_JSONL", "/mnt/data/sgsilva/datasets/pose3d/questions_3d_v3_v21_v4_v5_v6_v7_combined.jsonl"),
    "exercise_csv": os.environ.get("EXERCISE_CSV", "/home/sgsilva/vlm-post-training/aux_tasks/video_tasks/video_mcqa/generation/training/exercise_metadata.csv"),
    "processing_report": os.environ.get("PROCESSING_REPORT",
        "/mnt/data/shared/vlm/data/10k/processing_report.json"),
    "video_cache_dir": os.path.expanduser("~/.vlm_video_cache"),
    "port": 7862,
    "max_gallery_frames": 32,
    "default_fps": 8.0,
}

AMBER_COLORS = ["#D97706", "#92400E", "#FCD34D", "#F59E0B", "#B45309", "#FBBF24"]

APP_VIDEO_DATASETS_DIR = Path(os.environ.get(
    "APP_VIDEO_DATASETS_DIR",
    "/mnt/data/sgsilva/datasets/app_video_datasets",
))


def _scan_browse_datasets() -> List[Tuple[str, str]]:
    """Return sorted (label, path) pairs for all browse/inspect JSONLs in APP_VIDEO_DATASETS_DIR."""
    if not APP_VIDEO_DATASETS_DIR.exists():
        return []
    paths = sorted(
        APP_VIDEO_DATASETS_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [(p.stem, str(p)) for p in paths]


# ---------------------------------------------------------------------------
# 2. Cache management
# ---------------------------------------------------------------------------

_caches: list = []


def cacheable(maxsize=128):
    def decorator(func):
        cached = lru_cache(maxsize=maxsize)(func)
        _caches.append(cached)
        return cached
    return decorator


def clear_all_caches():
    for c in _caches:
        c.cache_clear()


# ---------------------------------------------------------------------------
# 3. Utilities
# ---------------------------------------------------------------------------

def empty_figure(message: str = "No data", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#999"))
    fig.update_layout(height=height, xaxis_visible=False, yaxis_visible=False,
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def tier_short(tier: str) -> str:
    return {"TIER_A_SINGLE_REP": "A", "TIER_B_COMPARATIVE": "B",
            "TIER_C_LONGITUDINAL": "C"}.get(tier, "?")


def tier_display(tier: str) -> str:
    return {"TIER_A_SINGLE_REP": "Tier A: Single Rep",
            "TIER_B_COMPARATIVE": "Tier B: Comparative",
            "TIER_C_LONGITUDINAL": "Tier C: Multi-rep",
            "TIER_D_BILATERAL": "Tier D: Bilateral"}.get(tier, tier)


def read_fps(directory: str) -> float:
    """Read fps.txt from directory, preferring per-rep over session fps.

    Lookup order:
      1. directory/fps.txt
      2. sibling repetitions/repetition_N/fps.txt
         (when called with cropped_repetitions/repetition_N — the cropped
         folder doesn't carry an fps.txt but the matching uncropped rep
         folder does, with rep-specific timing)
      3. session-wide images/fps.txt (fallback — coarser, averages all reps)
      4. CONFIG["default_fps"]

    NOTE: for video-id-keyed lookups prefer read_fps_for_video() which
    checks the JSONL metadata sidecar first.
    """
    d = Path(directory)
    # 1) directory itself
    fps_file = d / "fps.txt"
    if fps_file.exists():
        try:
            return float(fps_file.read_text().strip())
        except ValueError:
            pass
    # 2) sibling repetitions/repetition_N (per-rep, when caller passed
    # cropped_repetitions/repetition_N)
    if d.parent.name == "cropped_repetitions" and d.name.startswith("repetition_"):
        per_rep = d.parent.parent / "repetitions" / d.name / "fps.txt"
        if per_rep.exists():
            try:
                return float(per_rep.read_text().strip())
            except ValueError:
                pass
    # 3) session-wide images/fps.txt (averaged across all reps)
    images_fps = d.parent.parent / "images" / "fps.txt"
    if images_fps.exists():
        try:
            return float(images_fps.read_text().strip())
        except ValueError:
            pass
    return CONFIG["default_fps"]


def read_fps_for_video(video_id: str, fallback_dir: str = "") -> float:
    """FPS lookup keyed by video_id — sidecar preferred, fs-read fallback."""
    val = _sidecar_get(video_id, "fps")
    if val is not None:
        return float(val)
    if fallback_dir:
        return read_fps(fallback_dir)
    vd = resolve_video_dir(video_id)
    return read_fps(str(vd))


# ---------------------------------------------------------------------------
# 4. Data loading
# ---------------------------------------------------------------------------

# Module-level reasoning index — populated when a JSONL is loaded.
# Avoids threading it through Gradio filter cascade inputs.
_REASONING_INDEX: Dict = {}

# Module-level v6 comparison index — keyed by (video_id, template, joint_or_qhash)
# matching diff_v5_v6.py's _slot_key. Populated by load_v6_compare_index().
_V6_COMPARE_INDEX: Dict[Tuple[str, str, str], Dict] = {}

# Module-level video_path cache: video_id -> canonical directory path from metadata.
# Populated when a JSONL is loaded. Used by resolve_video_dir to find videos not in 10k/all.
_VIDEO_PATH_CACHE: Dict[str, str] = {}

# Module-level fps and flip caches: video_id -> value from metadata.
_VIDEO_FPS_CACHE: Dict[str, float] = {}
_VIDEO_FLIP_CACHE: Dict[str, bool] = {}

# 3D-feature sidecar cache: video_id → {rep_index_int: feat_dict}.
# Populated lazily by _load_pose3d_sidecar() on first access. Path is
# resolved by sniffing a small set of standard locations alongside the
# loaded JSONL (e.g. /mnt/data/sgsilva/datasets/pose3d/pose3d_features_q3d_v2_inprogress.jsonl).
_POSE3D_BY_VIDEO: Dict[str, Dict[int, dict]] = {}
_POSE3D_LOADED: bool = False
_POSE3D_SIDECAR_PATHS = [
    "/mnt/data/sgsilva/datasets/pose3d/pose3d_features_q3d_v3.jsonl",
    "/mnt/data/sgsilva/datasets/pose3d/pose3d_features_q3d_v21.jsonl",
    "/mnt/data/sgsilva/datasets/pose3d/pose3d_features_q3d_v2_inprogress.jsonl",
    "/mnt/data/sgsilva/datasets/pose3d/pose3d_features_q3d_v1_inprogress.jsonl",
    "/mnt/data/sgsilva/datasets/pose3d/_archive/pose3d_features_q3d_v1.jsonl",
]


# vo3d: oracle-obs ground-truth index for the cross-check column. Maps
# (session_id, rep_index) -> {q_index: oracle_answer_text}. Lazy-loaded from the
# oracle-obs categorical HF datasets (1105 + 1805) on first VO record render.
_ORACLE_OBS_BY_REP: Dict = {}
_ORACLE_OBS_LOADED: bool = False
_ORACLE_OBS_PATHS = [
    "/mnt/data/sgsilva/datasets/1105_oracle_obs_sft_train_categorical_emptythink",
    "/mnt/data/sgsilva/datasets/1805_oracle_obs_sft_train_categorical",
]
_NUM_LINE_RE = re.compile(r"^\s*(\d+)\.\s*(.*?)\s*$", re.M)

# Canonical schema (per-exercise vlm_observations) — so we can pair every oracle
# answer line with its question text, including questions we deferred.
_SCHEMA_BY_CODE: Dict = {}
_SCHEMA_LOADED: bool = False
_SCHEMA_PATHS = [
    "/home/sgsilva/vlm-post-training/visual_observations_categorical.json",
    "visual_observations_categorical.json",
]


def _load_schema() -> None:
    global _SCHEMA_LOADED
    if _SCHEMA_LOADED:
        return
    _SCHEMA_LOADED = True
    for p in _SCHEMA_PATHS:
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    _SCHEMA_BY_CODE.update(json.load(f))
                return
            except Exception:
                continue


def _schema_questions(exercise_code: str):
    """List of question dicts for an exercise code, or [] if unknown."""
    _load_schema()
    return (_SCHEMA_BY_CODE.get(str(exercise_code), {}) or {}).get("vlm_observations", [])


def _load_oracle_obs() -> None:
    """Lazy-load oracle-obs blocks, parsed into {(session,rep): {q_idx: answer}}."""
    global _ORACLE_OBS_LOADED
    if _ORACLE_OBS_LOADED:
        return
    _ORACLE_OBS_LOADED = True
    try:
        from datasets import load_from_disk
    except Exception:
        return
    for p in _ORACLE_OBS_PATHS:
        if not os.path.isdir(p):
            continue
        try:
            ds = load_from_disk(p)
        except Exception:
            continue
        for r in ds:
            key = (r.get("session_id"), int(r.get("rep_index", -1)))
            if key in _ORACLE_OBS_BY_REP:
                continue
            asst = next((m.get("content", "") for m in r.get("messages", [])
                         if m.get("role") == "assistant"), "")
            block = asst.split("</think>", 1)[-1]
            by_num = {}
            for num, body in _NUM_LINE_RE.findall(block):
                if body and not body.startswith("[VISUAL"):
                    by_num[int(num) - 1] = body.strip()  # 0-based q_index
            # keep the FULL raw oracle assistant content (unparsed) for display
            _ORACLE_OBS_BY_REP[key] = {"by_num": by_num, "raw": asst}


# vo3d: raw human PT annotation index (the actual ground truth the oracle was
# conditioned on). Maps (session_id, rep_index) -> dict of PT fields. Lazy-loaded
# from the human-annotation HF datasets (1105 + 1805, train + test splits).
_HUMAN_ANNOT_BY_REP: Dict = {}
_HUMAN_ANNOT_LOADED: bool = False
_HUMAN_ANNOT_PATHS = [
    "/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed/repetitions_train",
    "/mnt/data/shared/vlm/data/human_annotation_datasets/1105_not_reviewed/repetitions_test",
    "/mnt/data/shared/vlm/data/human_annotation_datasets/1805_not_reviewed/repetitions_train",
    "/mnt/data/shared/vlm/data/human_annotation_datasets/1805_not_reviewed/repetitions_test",
]
# Curated fields shown first, in this order; everything else in the source row is
# also captured and shown (we want as much info as possible). Bulky/redundant
# columns are skipped (frames + the rendered messages).
_HUMAN_ANNOT_FIELDS = ["analysis_of_movement", "therapist_feedback", "rom",
                       "severity_scores", "error_pattern", "effectiveness",
                       "injury_risk", "agreement_level"]
_HUMAN_ANNOT_SKIP = {"video_frames", "messages", "session_id", "rep_index"}


def _load_human_annot() -> None:
    """Lazy-load raw human PT annotations keyed by (session_id, rep_index)."""
    global _HUMAN_ANNOT_LOADED
    if _HUMAN_ANNOT_LOADED:
        return
    _HUMAN_ANNOT_LOADED = True
    try:
        from datasets import load_from_disk
    except Exception:
        return
    for p in _HUMAN_ANNOT_PATHS:
        if not os.path.isdir(p):
            continue
        try:
            ds = load_from_disk(p)
        except Exception:
            continue
        cols = [c for c in ds.column_names if c not in _HUMAN_ANNOT_SKIP]
        for r in ds:
            key = (r.get("session_id"), int(r.get("rep_index", -1)))
            if key in _HUMAN_ANNOT_BY_REP:
                continue
            # capture EVERY non-bulky column (not just the curated whitelist)
            _HUMAN_ANNOT_BY_REP[key] = {c: r.get(c) for c in cols}


def _load_pose3d_sidecar() -> None:
    """Lazy-load the 3D feature sidecar into _POSE3D_BY_VIDEO.

    Idempotent: subsequent calls are no-ops once a sidecar has been ingested.
    Silently skips missing files — the metrics renderer just shows an empty
    block when no 3D data is available for a given video.
    """
    global _POSE3D_LOADED
    if _POSE3D_LOADED:
        return
    for path in _POSE3D_SIDECAR_PATHS:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    vid = row.get("video_id")
                    per_rep = row.get("per_rep") or {}
                    if not vid or not per_rep:
                        continue
                    # First-write-wins across sidecar files: a video present
                    # in multiple datasets (v3, v21, …) keeps the rep data
                    # from the FIRST file listed in _POSE3D_SIDECAR_PATHS, so
                    # the default-dataset (v3) geometry isn't clobbered by a
                    # later file that selected a different rep for that video.
                    incoming = {int(k): v for k, v in per_rep.items()}
                    if vid in _POSE3D_BY_VIDEO:
                        # Merge any reps not already present, keep existing.
                        for k, v in incoming.items():
                            _POSE3D_BY_VIDEO[vid].setdefault(k, v)
                    else:
                        _POSE3D_BY_VIDEO[vid] = incoming
        except FileNotFoundError:
            continue
        except (json.JSONDecodeError, ValueError, OSError):
            continue
    _POSE3D_LOADED = True


def render_pose3d_metrics(video_id: str, restrict_to_reps: Optional[List[int]] = None) -> str:
    """Render a markdown block summarising 3D features for the given video.

    Returns "" when no 3D data is available so callers can safely concatenate.
    """
    _load_pose3d_sidecar()
    per_rep = _POSE3D_BY_VIDEO.get(video_id)
    if not per_rep:
        return ""

    reps = sorted(per_rep.items())
    if restrict_to_reps:
        wanted = set(int(r) for r in restrict_to_reps)
        reps = [(ri, f) for ri, f in reps if ri in wanted]
    if not reps:
        return ""

    lines = ["\n### 3D Features (SAM-3D-Body)\n"]

    # Session-level header row (pose class is invariant across reps for a
    # session; show it once if all reps agree, otherwise show per-rep).
    pose_classes = {f.get("pose_class") for _, f in reps}
    if len(pose_classes) == 1:
        pc = next(iter(pose_classes))
        lines.append(f"**Pose class**: `{pc}`\n")

    # Compact per-rep table — columns shown only when at least one rep has the
    # field, to avoid empty columns for unrelated exercises (e.g. trunk yaw
    # column on a Bridge session).
    has_yaw = any((f.get("trunk_axial_yaw") or {}).get("range_deg") is not None for _, f in reps)
    has_neck = any((f.get("neck_rotation") or {}).get("range_deg") is not None for _, f in reps)
    tl_any = any(f.get("trunk_lean") for _, f in reps)
    has_lat = any(((f.get("trunk_lean") or {}).get("lateral_max_deg") or 0) > 0 for _, f in reps)
    has_sag = any(((f.get("trunk_lean") or {}).get("max_deg") or 0) > 0 for _, f in reps)
    has_plane = any(f.get("motion_plane") for _, f in reps)
    has_active = any(f.get("active_side") for _, f in reps)
    has_asym = any(f.get("asymmetry_at_peak") for _, f in reps)

    headers = ["Rep"]
    if has_plane: headers.append("Motion plane")
    if has_active: headers.append("Active side")
    if has_yaw: headers.append("Axial yaw ROM")
    if has_yaw: headers.append("Yaw direction")
    if has_neck: headers.append("Neck rotation")
    if has_neck: headers.append("Neck direction")
    if has_lat: headers.append("Lateral lean")
    if has_lat: headers.append("Lateral dir.")
    if has_sag: headers.append("Sag. lean (from upright)")
    if has_sag: headers.append("Sag. sign")

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for ri, f in reps:
        row = [str(ri)]
        if has_plane:
            row.append(f.get("motion_plane") or "?")
        if has_active:
            row.append(f.get("active_side") or "?")
        ty = f.get("trunk_axial_yaw") or {}
        if has_yaw:
            r_deg = ty.get("range_deg")
            row.append(f"{r_deg:.1f}°" if r_deg is not None else "—")
            row.append(ty.get("direction") or "—")
        nk = f.get("neck_rotation") or {}
        if has_neck:
            r_deg = nk.get("range_deg")
            row.append(f"{r_deg:.1f}°" if r_deg is not None else "—")
            row.append(nk.get("direction") or "—")
        tl = f.get("trunk_lean") or {}
        if has_lat:
            lm = tl.get("lateral_max_deg")
            row.append(f"{lm:.1f}°" if lm is not None else "—")
            row.append(tl.get("lateral_direction") or "—")
        if has_sag:
            # max_deg is raw angle-from-gravity (~180° upright); show deviation.
            raw = tl.get("max_deg")
            row.append(f"{abs(raw - 180.0):.1f}°" if raw is not None else "—")
            row.append(tl.get("sign") or "—")
        lines.append("| " + " | ".join(row) + " |")

    # Asymmetry-at-peak summary (used by tier_c_coordination_3d). Show only
    # joints that appear in any rep.
    if has_asym:
        joint_kinds = sorted({
            jk for _, f in reps
            for jk, v in (f.get("asymmetry_at_peak") or {}).items()
            if v is not None
        })
        if joint_kinds:
            lines.append("")
            lines.append("**Asymmetry at peak (|L − R|, degrees):**\n")
            lines.append("| Rep | " + " | ".join(joint_kinds) + " |")
            lines.append("|" + "|".join(["---"] * (len(joint_kinds) + 1)) + "|")
            for ri, f in reps:
                a = f.get("asymmetry_at_peak") or {}
                row = [str(ri)] + [
                    (f"{a.get(jk):.1f}°" if a.get(jk) is not None else "—")
                    for jk in joint_kinds
                ]
                lines.append("| " + " | ".join(row) + " |")

    # Quality verdict row
    verdicts = [(ri, (f.get("quality") or {}).get("verdict")) for ri, f in reps]
    if any(v for _, v in verdicts):
        v_summary = ", ".join(f"rep {ri}: `{v or '?'}`" for ri, v in verdicts)
        lines.append("")
        lines.append(f"*Quality verdict: {v_summary}*")

    return "\n".join(lines)

# Per-exercise Tier-0 peak-angle distribution, populated at JSONL load.
# Used by render_verification to show "this rep's peak vs the exercise's
# peak-angle range across all samples in the loaded file".
# Key: exercise_code → list of peak_angle_degrees values.
_TIER0_PEAK_BY_EXERCISE: Dict[str, List[float]] = {}
# (exercise_code, primitive) → list of per-rep primitive values.
# Captures peak_angle_degrees AND velocity_ratio samples; lets the
# inspection app surface the cohort distribution alongside the current
# rep's value regardless of which Tier-0 primitive the sample used.
_TIER0_PRIMITIVE_BY_EXERCISE: Dict[Tuple[str, str], List[float]] = {}

def _redirect_features_to_questions(jsonl_path: str) -> str:
    """If the user pointed at a pose3d FEATURES sidecar (per-rep geometry,
    no questions), redirect to the sibling QUESTIONS file so the browser
    has something to render.

    Features files are named `pose3d_features_q3d_<ver>.jsonl`; the matching
    questions file is `questions_3d_<ver>.jsonl` in the same directory. If no
    sibling exists we leave the path unchanged (the caller raises a clear
    error). Backwards compatible: normal questions paths are returned as-is.
    """
    p = Path(jsonl_path)
    name = p.name
    if "pose3d_features_q3d_" in name:
        ver = name.replace("pose3d_features_q3d_", "").replace(".jsonl", "")
        sibling = p.parent / f"questions_3d_{ver}.jsonl"
        if sibling.is_file():
            return str(sibling)
    return jsonl_path


def load_jsonl_samples(jsonl_path: str) -> List[Dict]:
    global _VIDEO_PATH_CACHE, _VIDEO_FPS_CACHE, _VIDEO_FLIP_CACHE, _TIER0_PEAK_BY_EXERCISE, _TIER0_PRIMITIVE_BY_EXERCISE
    # A pose3d features sidecar has no questions — redirect to its sibling
    # questions file when one exists.
    jsonl_path = _redirect_features_to_questions(jsonl_path)
    samples = []
    path_cache, fps_cache, flip_cache = {}, {}, {}
    tier0_peak_by_exercise: Dict[str, List[float]] = {}
    tier0_primitive_by_exercise: Dict[Tuple[str, str], List[float]] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                msgs = sample.get("messages", [])
                if isinstance(msgs, str):
                    sample["messages"] = json.loads(msgs)
                samples.append(sample)
                meta = sample.get("metadata", {})
                # video_id may live in metadata (MCQA) or only top-level
                # session_id (vo3d / oracle-obs flat format).
                vid = meta.get("video_id", "") or sample.get("session_id", "")
                if vid:
                    if meta.get("video_path"):
                        path_cache[vid] = meta["video_path"]
                    if meta.get("fps") is not None:
                        fps_cache[vid] = float(meta["fps"])
                    # need_to_flip: metadata first, else TOP-LEVEL (vo3d/oracle-obs
                    # put it at the row top level, not inside metadata).
                    flip_val = meta.get("need_to_flip")
                    if flip_val is None:
                        flip_val = sample.get("need_to_flip")
                    if flip_val is not None:
                        flip_cache[vid] = bool(flip_val)
                    # Also pick up top-level fps from the row
                    if sample.get("fps") is not None and vid not in fps_cache:
                        fps_cache[vid] = float(sample["fps"])
                # Tier-0 peak-angle distribution per exercise (populated when
                # the sample carries tier0_peak_angle_degrees in its
                # verification block; lets the inspection app surface the
                # cohort range alongside the current sample's value).
                ver = meta.get("verification") or {}
                ec = meta.get("exercise_code", "")
                peak = ver.get("tier0_peak_angle_degrees")
                if peak is not None and ec:
                    tier0_peak_by_exercise.setdefault(ec, []).append(float(peak))
                # Generic primitive distribution (amplitude + control).
                # Keyed by (exercise_code, primitive_name) so amplitude and
                # control axes do not collide on the same exercise.
                primitive = ver.get("tier0_primitive")
                if primitive and ec:
                    pv_field = {
                        "peak_angle_degrees": "tier0_peak_angle_degrees",
                        "max_angle_degrees": "tier0_max_angle_degrees",
                        "velocity_ratio": "tier0_velocity_ratio",
                    }.get(primitive)
                    pv = ver.get(pv_field) if pv_field else None
                    if pv is not None:
                        tier0_primitive_by_exercise.setdefault((ec, primitive), []).append(float(pv))
    # Guard: a file with rows but no `messages` on any of them is not a
    # questions file (most likely a pose3d FEATURES sidecar loaded directly,
    # with no sibling questions file to redirect to). Fail loudly rather
    # than rendering a browser full of blank questions.
    if samples and not any(s.get("messages") for s in samples):
        has_per_rep = any(s.get("per_rep") for s in samples)
        hint = (" This looks like a pose3d FEATURES sidecar (per-rep "
                "geometry, no questions) — load the matching "
                "`questions_3d_<ver>.jsonl` instead.") if has_per_rep else ""
        raise ValueError(
            f"No `messages` found in any of the {len(samples)} rows of "
            f"{Path(jsonl_path).name} — this is not a questions file.{hint}")

    _VIDEO_PATH_CACHE = path_cache
    _VIDEO_FPS_CACHE = fps_cache
    _TIER0_PEAK_BY_EXERCISE = tier0_peak_by_exercise
    _TIER0_PRIMITIVE_BY_EXERCISE = tier0_primitive_by_exercise
    _VIDEO_FLIP_CACHE = flip_cache

    # Sort: primary = exercise_code (numeric), secondary = tier (A < B < C < D)
    _TIER_ORDER = {
        "TIER_A_SINGLE_REP": 0, "TIER_B_COMPARATIVE": 1,
        "TIER_C_LONGITUDINAL": 2, "TIER_D_BILATERAL": 3,
    }
    def _sort_key(s):
        meta = s.get("metadata", {})
        code = meta.get("exercise_code") or meta.get("video_id", "").split("_")[0]
        try:
            code_int = int(code)
        except (ValueError, TypeError):
            code_int = 0
        tier_int = _TIER_ORDER.get(meta.get("difficulty_tier", ""), 99)
        return (code_int, tier_int)
    samples.sort(key=_sort_key)

    return samples


def load_reasoning_index(jsonl_path: str) -> Dict:
    """Load a reasoning index built by build_reasoning_index.py.
    Returns dict keyed by (key_frame, key_text) -> entry dict.
    Also populates the module-level _REASONING_INDEX for use in filter callbacks."""
    global _REASONING_INDEX
    index_path = Path(jsonl_path).parent / (Path(jsonl_path).stem.replace("qa_samples", "reasoning_index") + ".jsonl")
    if index_path == Path(jsonl_path) or not index_path.exists():
        logger.info(f"No reasoning index found at {index_path}")
        _REASONING_INDEX = {}
        return {}
    index = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                index[(e["key_frame"], e["key_text"])] = e
    logger.info(f"Loaded {len(index)} reasoning entries from {index_path}")
    _REASONING_INDEX = index
    return index


def _v6_slot_key(sample: Dict) -> Tuple[str, str, str]:
    """Mirror of diff_v5_v6.py::_slot_key for app-side lookups."""
    import hashlib as _h
    meta = sample.get("metadata", {})
    vid = str(meta.get("video_id", ""))
    tmpl = str(meta.get("question_template", ""))
    verification = meta.get("verification", {})
    joint = (
        verification.get("joint")
        or verification.get("compensatory_joint")
        or meta.get("joint")
        or ""
    )
    if not joint:
        q = sample.get("question") or sample.get("question_text") or ""
        # Fallback: hash question text from messages[0].content if needed
        if not q:
            msgs = sample.get("messages", [])
            if msgs and isinstance(msgs, list):
                first = msgs[0]
                if isinstance(first, dict):
                    c = first.get("content", "")
                    q = c if isinstance(c, str) else json.dumps(c)
        joint = "q:" + _h.sha1(str(q).encode()).hexdigest()[:10]
    return (vid, tmpl, str(joint))


def load_v6_compare_index(jsonl_path: str) -> int:
    """Load a v6 JSONL into the module-level comparison index. Returns count."""
    global _V6_COMPARE_INDEX
    if not jsonl_path or not Path(jsonl_path).exists():
        _V6_COMPARE_INDEX = {}
        return 0
    idx: Dict[Tuple[str, str, str], Dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx[_v6_slot_key(d)] = d
    _V6_COMPARE_INDEX = idx
    return len(idx)


def render_v6_comparison(sample: Dict) -> str:
    """Render markdown comparing the current sample to the matching one in the
    comparison JSONL (typically the other dataset version).

    The current sample is the "primary" (whatever you loaded). The comparison
    sample is from the loaded comparison file. Labels are side-agnostic so it
    works whether you loaded v5 or v6 as primary.
    """
    if not _V6_COMPARE_INDEX:
        return ""
    key = _v6_slot_key(sample)
    other = _V6_COMPARE_INDEX.get(key)
    audit = sample.get("v6_audit", []) or []
    audit_md = ""
    if audit:
        audit_md = f"_v6_audit tags on this sample:_ `{', '.join(audit)}`\n\n"

    if other is None:
        return f"### Side-by-side comparison\n\n{audit_md}**No matching slot in comparison JSONL** — this sample is unique to the loaded primary file."

    # Two samples are SAME when the correct-answer text matches AND the set of
    # choices matches. Question-stem wording can differ (template chosen by
    # RNG) and choice ordering is shuffled — both should be ignored when
    # deciding "did this question actually change".
    def _qtext(d):
        msgs = d.get("messages")
        if isinstance(msgs, list) and len(msgs) >= 2:
            content = msgs[1].get("content") if isinstance(msgs[1], dict) else None
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        return str(block["text"]).strip()
            elif isinstance(content, str):
                return content.strip()
        return (d.get("question") or d.get("question_text") or "").strip()

    def _choices(d):
        meta = d.get("metadata", {}) or {}
        out = []
        for c in (meta.get("choices") or []):
            if isinstance(c, dict):
                out.append(str(c.get("text") or c.get("answer") or "").strip())
            else:
                out.append(str(c).strip())
        return out

    def _correct(d):
        meta = d.get("metadata", {}) or {}
        if meta.get("correct_text"):
            return str(meta["correct_text"]).strip()
        if "correct_index" in meta and isinstance(meta.get("choices"), list):
            idx = meta["correct_index"]
            if isinstance(idx, int) and 0 <= idx < len(meta["choices"]):
                c = meta["choices"][idx]
                return str(c.get("text") if isinstance(c, dict) else c).strip()
        ans = meta.get("correct_answer")
        if ans and isinstance(meta.get("choices"), list) and isinstance(ans, str) and len(ans) == 1:
            idx = ord(ans.upper()) - ord("A")
            if 0 <= idx < len(meta["choices"]):
                c = meta["choices"][idx]
                return str(c.get("text") if isinstance(c, dict) else c).strip()
        return str(ans or "").strip()

    a_q, b_q = _qtext(sample), _qtext(other)
    a_a, b_a = _correct(sample), _correct(other)
    a_choices, b_choices = _choices(sample), _choices(other)
    same = a_a == b_a and frozenset(a_choices) == frozenset(b_choices)
    status = "**SAME** (same choices + correct, possibly different template wording)" if same else "**CHANGED**"
    md = f"### Side-by-side comparison — {status}\n\n{audit_md}"
    if not same:
        if a_q != b_q:
            md += f"**Primary question:** {a_q}\n\n**Comparison question:** {b_q}\n\n"
        if a_a != b_a:
            md += f"**Primary correct:** {a_a}\n\n**Comparison correct:** {b_a}\n\n"
        elif frozenset(a_choices) != frozenset(b_choices):
            md += f"**Primary choices:** {' | '.join(a_choices)}\n\n**Comparison choices:** {' | '.join(b_choices)}\n\n"
    return md


def lookup_reasoning(sample: Dict, reasoning_index: Dict = None) -> Optional[Dict]:
    """Find the reas2 reasoning entry for a qa_samples row.
    Falls back to module-level _REASONING_INDEX if reasoning_index is empty."""
    idx = reasoning_index if reasoning_index else _REASONING_INDEX
    if not idx:
        return None
    vf = sample.get("video_frames") or []
    key_frame = vf[0] if vf else ""
    msgs = sample.get("messages", [])
    key_text = msgs[1].get("content", "") if len(msgs) > 1 else ""
    if isinstance(key_text, list):
        for item in key_text:
            if isinstance(item, dict) and "text" in item:
                key_text = item["text"]
                break
    return idx.get((key_frame, key_text))


def _normalize_error_label(raw: str) -> str:
    """Normalize verbose CSV error_category into a clean group label."""
    if not raw:
        return ""
    cat = re.sub(r"^\d+\.\s*", "", raw)  # strip leading numbering
    low = cat.lower()
    if any(k in low for k in ("insufficient", "incomplete", "range of motion")):
        for part, label in [("shoulder", "Insufficient ROM (shoulder)"),
                            ("hip", "Insufficient ROM (hip)"),
                            ("knee", "Insufficient ROM (knee)"),
                            ("trunk", "Insufficient ROM (trunk)"),
                            ("elbow", "Insufficient ROM (elbow)"),
                            ("wrist", "Insufficient ROM (wrist/hand)"),
                            ("finger", "Insufficient ROM (wrist/hand)"),
                            ("scapul", "Insufficient ROM (scapular)"),
                            ("plantar", "Insufficient ROM (ankle)"),
                            ("ankle", "Insufficient ROM (ankle)"),
                            ("dorsi", "Insufficient ROM (ankle)"),
                            ("thoracic", "Insufficient ROM (thoracic)")]:
            if part in low:
                return label
        return "Insufficient ROM (general)"
    if any(k in low for k in ("momentum", "swinging", "bouncing", "speed")):
        return "Momentum/swinging"
    if any(k in low for k in ("lack of", "poor control", "jerky", "rapid",
                               "uncontrolled", "abrupt", "shaky", "choppy")):
        return "Poor control/jerky"
    if any(k in low for k in ("asymmetr", "uneven", "one side", "lateral shift")):
        return "Asymmetry"
    if any(k in low for k in ("pelvic", "pelvis")):
        return "Pelvic drop/rotation"
    if any(k in low for k in ("knee", "valgus", "varus")):
        return "Knee misalignment"
    if any(k in low for k in ("fatigue", "loss of precision", "tiring")):
        return "Fatigue"
    return "Other"


def build_index(samples: List[Dict]) -> Dict[str, Dict[str, List[int]]]:
    """Build filter indices: tier/joint/exercise/template/error_label -> [sample indices]."""
    index: Dict[str, Dict[str, List[int]]] = {
        "tiers": {}, "joints": {}, "exercises": {}, "exercise_ids": {}, "templates": {},
        "error_labels": {}, "body_regions": {}, "kp_sources": {}, "categories": {}, "exercise_types": {},
        "frames_sources": {}, "diff_status": {}, "camera_perspectives": {},
        "source_datasets": {},
    }

    # Pre-build code→name lookup (O(1) per sample instead of DataFrame scan)
    code_to_name: Dict[str, str] = {}
    if not EXERCISE_DF.empty:
        for _, row in EXERCISE_DF.iterrows():
            code_to_name[row.get("exercise_code", "")] = row.get("exercise_name", "unknown")

    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        tier = meta.get("difficulty_tier", "unknown")
        index["tiers"].setdefault(tier, []).append(i)

        template = meta.get("question_template", "unknown")
        index["templates"].setdefault(template, []).append(i)

        joint = meta.get("verification", {}).get("joint") or meta.get("verification", {}).get("compensatory_joint") or "unknown"
        index["joints"].setdefault(joint, []).append(i)

        # Error label (from verification.error_category, normalized)
        raw_error = meta.get("verification", {}).get("error_category", "")
        if raw_error:
            error_label = _normalize_error_label(raw_error)
            index["error_labels"].setdefault(error_label, []).append(i)

        # Body region
        body_region = meta.get("body_region", "")
        if body_region:
            index["body_regions"].setdefault(body_region, []).append(i)

        # Keypoint / feature source (sam3dbody / vitpose / blazepose).
        # Sample-aware: SAM-3D-Body for mesh/pose-class 3D features, else
        # the per-video VitPose/BlazePose resolution. Backwards compatible
        # — non-3D samples bucket exactly as before.
        kp_src = _resolve_kp_source_for_sample(s)
        index["kp_sources"].setdefault(kp_src, []).append(i)

        # Source dataset (set when browsing a combined file, e.g.
        # questions_3d_v3_plus_v21.jsonl). Absent on single-dataset files —
        # those bucket under "(single)" so the filter only offers "All".
        src_ds = meta.get("_source_dataset") or "(single)"
        index["source_datasets"].setdefault(src_ds, []).append(i)

        # Category (bilateral_category / category field)
        cat = meta.get("category") or meta.get("bilateral_category") or ""
        if cat:
            index["categories"].setdefault(cat, []).append(i)

        # Exercise type (ROM / Hold / RNR)
        ex_type = meta.get("exercise_type") or ""
        if ex_type:
            index["exercise_types"].setdefault(ex_type, []).append(i)

        # Frames source (cropped vs uncropped). Default to "cropped" for pre-v5
        # samples that predate the frames_source stamp.
        fr_src = meta.get("frames_source") or "cropped"
        index["frames_sources"].setdefault(fr_src, []).append(i)

        # Camera perspective (frontal/lateral). Set on samples generated from
        # v10+ JSONLs (or backfilled from CSV).
        cam = meta.get("camera_perspective") or ""
        if cam:
            index["camera_perspectives"].setdefault(cam, []).append(i)

        # Look up the matching slot in the comparison-JSONL index. Used by the
        # diff_status block below to detect Added / Changed / Same.
        comp = None
        if _V6_COMPARE_INDEX:
            try:
                key = _v6_slot_key(s)
                comp = _V6_COMPARE_INDEX.get(key)
            except Exception:
                comp = None

        # Diff status against the comparison JSONL. Independent from v6_audit:
        #   Added   — primary sample has no matching slot in comparison
        #   Removed — comparison has the slot but primary doesn't (only meaningful
        #             from the comparison's POV; we never flag this for primary)
        #   Changed — both exist but (correct_text, choice-set) differ
        #   Same    — both exist and (correct_text, choice-set) match
        # Question-stem wording and choice ordering are ignored.
        if _V6_COMPARE_INDEX:
            if comp is None:
                diff_status = "Added"
            else:
                try:
                    p_meta = s.get("metadata", {}) or {}
                    c_meta = comp.get("metadata", {}) or {}

                    def _ct(m):
                        if m.get("correct_text"):
                            return str(m["correct_text"]).strip()
                        if "correct_index" in m and isinstance(m.get("choices"), list):
                            ci = m["correct_index"]
                            if isinstance(ci, int) and 0 <= ci < len(m["choices"]):
                                cc = m["choices"][ci]
                                return str(cc.get("text") if isinstance(cc, dict) else cc).strip()
                        return ""

                    def _chset(m):
                        out = []
                        for cc in (m.get("choices") or []):
                            if isinstance(cc, dict):
                                out.append(str(cc.get("text") or cc.get("answer") or "").strip())
                            else:
                                out.append(str(cc).strip())
                        return frozenset(out)

                    if _ct(p_meta) != _ct(c_meta) or _chset(p_meta) != _chset(c_meta):
                        diff_status = "Changed"
                    else:
                        diff_status = "Same"
                except Exception:
                    diff_status = "Changed"
            index["diff_status"].setdefault(diff_status, []).append(i)

        # Resolve exercise name from pre-built lookup
        exercise = meta.get("exercise_name", "unknown")
        vid = meta.get("video_id", "")
        code = meta.get("exercise_code") or (vid.split("_")[0] if vid else "")
        if code and code_to_name:
            exercise = code_to_name.get(code, exercise)
        index["exercises"].setdefault(exercise, []).append(i)

        # exercise_id index (numeric code as string, e.g. "10052")
        if code:
            index["exercise_ids"].setdefault(code, []).append(i)

    # Build per-video question numbering: sample_index -> (q_num, total)
    video_samples: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        vid = s.get("metadata", {}).get("video_id", "")
        video_samples.setdefault(vid, []).append(i)
    q_numbering: Dict[int, Tuple[int, int]] = {}
    for vid, indices in video_samples.items():
        for q_num, idx in enumerate(indices, 1):
            q_numbering[idx] = (q_num, len(indices))
    index["q_numbering"] = q_numbering

    return index


def get_video_label_cached(video_id: str) -> str:
    """Cached wrapper for get_video_label_summary. Avoids repeated filesystem reads."""
    if not hasattr(get_video_label_cached, "_cache"):
        get_video_label_cached._cache = {}
    cache = get_video_label_cached._cache
    if video_id not in cache:
        cache[video_id] = get_video_label_summary(video_id)
    return cache[video_id]


def get_filtered_indices(
    index: Dict, total: int,
    tier: str = "All", joint: str = "All", exercise: str = "All",
    exercise_id: str = "All",
    movement_label: str = "All", samples: List[Dict] = None,
    error_type: str = "All", body_region: str = "All",
    kp_source: str = "All", template: str = "All", category: str = "All",
    exercise_type: str = "All", frames_source: str = "All",
    annotation: str = "All", annotations: Optional[dict] = None,
    diff_status: str = "All",
    camera_perspective: str = "All",
    annotated_by: str = "All",
    judge_pass1: str = "All",
    judge_pass2: str = "All",
    salvage_origin: str = "All",
    geometry_assessment: str = "All",
    source_dataset: str = "All",
) -> List[int]:
    all_set = set(range(total))

    # tier can be a list (multiselect) or a string
    tier_selections = tier if isinstance(tier, list) else [tier] if tier else []
    # Reverse-map readable labels back to enum keys
    _label_to_tier = {
        "Tier A (single rep)": "TIER_A_SINGLE_REP",
        "Tier B (comparative)": "TIER_B_COMPARATIVE",
        "Tier C (multi-rep)": "TIER_C_LONGITUDINAL",
        "Tier D (bilateral)": "TIER_D_BILATERAL",
    }
    tier_selections = [_label_to_tier.get(t, t) for t in tier_selections if t]
    if tier_selections:
        tier_set = set()
        for t in tier_selections:
            if t in index["tiers"]:
                tier_set |= set(index["tiers"][t])
            elif t in index.get("templates", {}):
                tier_set |= set(index["templates"][t])
        if tier_set:
            all_set &= tier_set
    if joint != "All" and joint in index["joints"]:
        all_set &= set(index["joints"][joint])
    if exercise != "All" and exercise in index["exercises"]:
        all_set &= set(index["exercises"][exercise])

    if exercise_id != "All" and exercise_id in index.get("exercise_ids", {}):
        all_set &= set(index["exercise_ids"][exercise_id])

    if movement_label != "All" and samples:
        label_set = set()
        for i in all_set:
            vid = samples[i].get("metadata", {}).get("video_id", "")
            if get_video_label_cached(vid) == movement_label:
                label_set.add(i)
        all_set = label_set

    if error_type != "All" and error_type in index.get("error_labels", {}):
        all_set &= set(index["error_labels"][error_type])

    if body_region != "All" and body_region in index.get("body_regions", {}):
        all_set &= set(index["body_regions"][body_region])

    if kp_source != "All" and kp_source in index.get("kp_sources", {}):
        all_set &= set(index["kp_sources"][kp_source])

    if template != "All" and template in index.get("templates", {}):
        all_set &= set(index["templates"][template])

    if category != "All" and category in index.get("categories", {}):
        all_set &= set(index["categories"][category])

    if exercise_type != "All" and exercise_type in index.get("exercise_types", {}):
        all_set &= set(index["exercise_types"][exercise_type])

    if frames_source != "All" and frames_source in index.get("frames_sources", {}):
        all_set &= set(index["frames_sources"][frames_source])

    if camera_perspective != "All" and camera_perspective in index.get("camera_perspectives", {}):
        all_set &= set(index["camera_perspectives"][camera_perspective])

    if diff_status != "All" and diff_status in index.get("diff_status", {}):
        all_set &= set(index["diff_status"][diff_status])

    if source_dataset != "All" and source_dataset in index.get("source_datasets", {}):
        all_set &= set(index["source_datasets"][source_dataset])

    # Annotation filter: match by generation_timestamp key against annotations dict.
    if annotation != "All" and samples and annotations is not None:
        keep = set()
        if annotation == "Unrated":
            for i in all_set:
                ts = samples[i].get("metadata", {}).get("generation_timestamp", "")
                if ts not in annotations:
                    keep.add(i)
        else:
            for i in all_set:
                ts = samples[i].get("metadata", {}).get("generation_timestamp", "")
                ann = annotations.get(ts)
                if ann and ann.get("rating") == annotation:
                    keep.add(i)
        all_set = keep

    # Annotated-by filter: keep samples whose annotation was authored by
    # the selected user (or "(unknown)" for legacy entries with no author).
    if annotated_by != "All" and samples and annotations is not None:
        keep = set()
        for i in all_set:
            ts = samples[i].get("metadata", {}).get("generation_timestamp", "")
            ann = annotations.get(ts)
            if not ann:
                continue
            author = ann.get("author") or "(unknown)"
            if author == annotated_by:
                keep.add(i)
        all_set = keep

    # Judge-verdict filters — Pass 1 and Pass 2 are independent dropdowns
    # so the user can ask "show me samples Pass 2 calls a side mismatch"
    # without that interacting with Pass 1, and vice versa.
    # When both are non-default the filters AND together (sample must match
    # both passes' selected verdict).
    _failure_set = {"parse_failed", "error", "pending"}

    def _matches_pass1(sample: Dict, sel: str) -> bool:
        if sel == "All":
            return True
        cats = [v["category"] for v in _sample_judge_verdicts(sample)]
        if sel == "Any non-ok":
            return any(c != "ok" for c in cats)
        if sel == "No verdict":
            return not cats
        return sel in cats

    def _matches_pass2(sample: Dict, sel: str) -> bool:
        if sel == "All":
            return True
        verdicts = [v["verdict"] for v in _sample_pass2_verdicts(sample)]
        if sel == "Any non-ok":
            return any(v != "side_matches" and v != "side_not_applicable"
                       for v in verdicts)
        if sel == "No verdict":
            return not verdicts
        return sel in verdicts

    # Salvage origin: filter by whether the sample carries the
    # `metadata.salvaged_human` provenance block (added by the human-salvage
    # pipeline; see scripts/build_human_salvage.py).
    if salvage_origin != "All" and samples:
        def _is_salvaged(i: int) -> bool:
            md = (samples[i].get("metadata") or {})
            return bool(md.get("salvaged_human"))
        if salvage_origin == "Salvaged (human Good quality, prior version)":
            all_set = {i for i in all_set if _is_salvaged(i)}
        elif salvage_origin == "Net-new (current version)":
            all_set = {i for i in all_set if not _is_salvaged(i)}
        elif salvage_origin == "Salvaged & needs reverification":
            all_set = {
                i for i in all_set
                if (samples[i].get("metadata") or {}).get("salvaged_human", {}).get("needs_reverification") is True
            }
        elif salvage_origin == "Salvaged & confirmed by sgsilva":
            all_set = {
                i for i in all_set
                if (samples[i].get("metadata") or {}).get("salvaged_human", {}).get("needs_reverification") is False
            }

    if (judge_pass1 != "All" or judge_pass2 != "All") and samples:
        all_set = {
            i for i in all_set
            if _matches_pass1(samples[i], judge_pass1)
            and _matches_pass2(samples[i], judge_pass2)
        }

    # Geometry-assessment filter: surfaces metadata.geometry_assessment
    # injected by the abduction-sagittal-chain audit (qa_abduction_sagittal_chain_with_assessment.jsonl).
    # MODE_2_DROP = wrong chain (sagittal applied to an abduction exercise) — calc invalid.
    # MODE_1_KEEP = wrong joint but calc valid (knee flexion on hip abduction, etc.).
    if geometry_assessment != "All" and samples:
        keep = set()
        for i in all_set:
            md = samples[i].get("metadata") or {}
            ga = md.get("geometry_assessment")
            if geometry_assessment == "MODE_1_KEEP" and ga == "MODE_1_KEEP":
                keep.add(i)
            elif geometry_assessment == "MODE_2_DROP" and ga == "MODE_2_DROP":
                keep.add(i)
            elif geometry_assessment == "Any (has assessment)" and ga in ("MODE_1_KEEP", "MODE_2_DROP", "UNCLEAR"):
                keep.add(i)
            elif geometry_assessment == "Unassessed" and not ga:
                keep.add(i)
        all_set = keep

    return sorted(all_set)


def parse_sample(sample: Dict) -> Dict[str, Any]:
    """Extract display-ready fields from a JSONL sample."""
    msgs = sample.get("messages", [])
    meta = sample.get("metadata", {})

    video_path = ""
    question_text = ""
    if len(msgs) > 1:
        user_content = msgs[1].get("content", [])
        if isinstance(user_content, list):
            for item in user_content:
                if isinstance(item, dict):
                    if "video" in item:
                        video_path = item["video"]
                    if "text" in item:
                        question_text = item["text"]
        elif isinstance(user_content, str):
            # Production format: plain string content (no video dict)
            question_text = user_content

    correct_letter = msgs[2].get("content", "?") if len(msgs) > 2 else "?"
    # Strip reasoning traces from assistant content — keep only the answer letter
    if isinstance(correct_letter, str) and "<think>" in correct_letter:
        import re as _re
        correct_letter = _re.sub(r"<think>.*?</think>\s*", "", correct_letter, flags=_re.DOTALL).strip()

    return {
        "video_path": video_path,
        # video_id lives in metadata for MCQA, but flat oracle-obs/vo3d/binary rows
        # only carry top-level session_id. Fall back to it so the flip/fps/path
        # caches (keyed by `meta.video_id or session_id` in load_jsonl_samples)
        # are actually hit — otherwise need_to_flip is silently lost (shows unflipped).
        "video_id": meta.get("video_id", "") or sample.get("session_id", ""),
        "question_text": question_text,
        "correct_letter": correct_letter,
        "correct_index": meta.get("correct_index", -1),
        "correct_text": meta.get("correct_text", ""),
        "choices": meta.get("choices", []),
        "tier": meta.get("difficulty_tier", ""),
        "joint": meta.get("verification", {}).get("joint", ""),
        "exercise_name": meta.get("exercise_name", ""),
        "template": meta.get("question_template", ""),
        "verification": meta.get("verification", {}),
        "distractor_sources": meta.get("distractor_sources", []),
        "rep_comparison": meta.get("rep_comparison", {}),
        "metadata": meta,
        # Canonical frame list from production dataset — exact cropped frames for this question.
        # Present when the JSONL was built from mcqa_video_0804; None for older files.
        "video_frames": sample.get("video_frames") or [],
        "fps": sample.get("fps") or meta.get("fps"),
        # vo3d Visual-Observations format: one full [VISUAL OBSERVATIONS] block per
        # rep (not per-question MCQA). Detected by metadata.answers[] presence.
        "vo_block": (correct_letter if isinstance(correct_letter, str)
                     and "[VISUAL OBSERVATIONS]" in correct_letter else ""),
        "vo_answers": meta.get("answers") if isinstance(meta.get("answers"), list) else None,
    }


# ── Annotation persistence ──────────────────────────────────────────────
ANNOTATIONS_DIR = Path(CONFIG.get("annotations_dir", str(DATA_DIR / "annotations")))


def _annotations_path(jsonl_path: str) -> Path:
    """Derive annotation file path from JSONL path."""
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return ANNOTATIONS_DIR / f"{Path(jsonl_path).stem}_annotations.json"


# ---------------------------------------------------------------------------
# Judge-verdict cache: maps (video_id, rep_index) -> {"category", "confidence",
# "evidence"}. Loaded from judge_flags_*.jsonl files when JSONL is loaded.
# ---------------------------------------------------------------------------
JUDGE_FLAGS_CANDIDATE_PATHS = [
    Path("/home/sgsilva/tmp/judge_debug_5/judge_flags_35.jsonl"),
    Path("/mnt/data/sgsilva/tmp/judge_flags_v13_1.jsonl"),
    Path("/mnt/data/sgsilva/tmp/judge_flags_calibration.jsonl"),
    # v14 in-scope judge runs (8 codes × 27 samples). Loader order matters:
    # later files override earlier ones for the same (video, rep) key, so put
    # the most recent prompt revision last (Phase D wider + parse-fix).
    # v14 judge runs were moved 2026-05-14 from /mnt/data/sgsilva/tmp/ →
    # /mnt/data/sgsilva/datasets/qa_versions/judge_outputs/
    Path("/mnt/data/sgsilva/datasets/qa_versions/judge_outputs/judge_v14_in_scope.jsonl"),     # narrow Phase D
    Path("/mnt/data/sgsilva/datasets/qa_versions/judge_outputs/judge_v14_widefix.jsonl"),      # wider Phase D
    Path("/mnt/data/sgsilva/datasets/qa_versions/judge_outputs/judge_v14_widefix_v2.jsonl"),   # wider + parse-fix
    Path("/mnt/data/sgsilva/datasets/qa_versions/judge_outputs/judge_v14_widefix_v3.jsonl"),   # wider + parse-fix + all prompt fixes
    # Re-judge of stale wrong_side_for_question flags from judge_flags_35.jsonl
    # on swap=yes codes (those used the pre-Phase-D anatomical-only legend and
    # mis-classify under the current camera-relative one).
    Path("/mnt/data/sgsilva/datasets/qa_versions/judge_outputs/judge_v14_stale35_rejudge.jsonl"),
    # Full v14 corpus judge run (Qwen3.5-397B on worker-22 with current
    # prompt: wider Phase D + parse-fix + all remediation). 3435 verdicts
    # spanning every sample in v14, including --include-multirep so Tier B
    # and Tier C samples each produce one verdict per referenced rep.
    Path("/mnt/data/sgsilva/datasets/qa_versions/judge_outputs/judge_v14_full.jsonl"),
    # Salvage-only judge run on the 83 unique salvage videos in v14_added
    # (264 verdicts). Current prompt. Highest priority for the keys it covers
    # so the app shows the most-recent verdict on salvaged samples.
    Path("/mnt/data/sgsilva/datasets/qa_versions/judge_outputs/judge_v14_added_salvage.jsonl"),  # current
]
JUDGE_CATEGORIES_KNOWN = [
    "ok",
    "patient_not_performing",
    "multi_rep_in_one_rep",
    "wrong_exercise",
    "pose_estimation_wrong_subject",
    "wrong_side_for_question",
    "parse_failed",
    "error",
    "pending",
]


def _load_judge_flags() -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Merge every judge_flags JSONL we can find into one (video, rep) → record map.
    Later files override earlier ones for the same key (so `judge_flags_v13_1`
    wins over the calibration set when both exist)."""
    merged: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path in JUDGE_FLAGS_CANDIDATE_PATHS:
        if not path.is_file():
            continue
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    vid = d.get("video_id", "")
                    rep = d.get("rep_index")
                    if not vid or not isinstance(rep, int):
                        continue
                    # Default category to "pending" — never to "ok" — so a
                    # missing `category` field can't masquerade as a clean
                    # verdict. (Real verdicts always set the field.)
                    merged[(vid, rep)] = {
                        "category": d.get("category", "pending"),
                        "confidence": float(d.get("confidence") or 0.0),
                        "evidence": d.get("evidence", ""),
                        "error": d.get("error", ""),
                        "source": path.name,
                    }
        except OSError:
            continue
    return merged


_JUDGE_FLAGS: Dict[Tuple[str, int], Dict[str, Any]] = _load_judge_flags()


# ---------------------------------------------------------------------------
# Pass 2 (side-mismatch) cache. Key: (video_id, rep_index, generation_timestamp)
# because the same (video, rep) clip can be asked about by multiple samples
# (one for left side, one for right side, etc.) and each sample gets its own
# Pass-2 verdict.
# ---------------------------------------------------------------------------
JUDGE_PASS2_CANDIDATE_PATHS = [
    Path("/home/sgsilva/tmp/judge_debug_5/judge_pass2_flags_35.jsonl"),
    Path("/mnt/data/sgsilva/tmp/judge_pass2_flags_v13_1.jsonl"),
]
PASS2_VERDICTS_KNOWN = [
    "side_matches",
    "side_mismatch",
    "side_not_applicable",
    "parse_failed",
    "error",
    "pending",
]


def _load_judge_pass2_flags() -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """Load Pass-2 (side-mismatch) verdicts. Keyed by
    (video_id, rep_index, generation_timestamp). Empty timestamp falls
    back to "" — still unique within (video_id, rep_index) for samples
    that lack a timestamp."""
    merged: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for path in JUDGE_PASS2_CANDIDATE_PATHS:
        if not path.is_file():
            continue
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    vid = d.get("video_id", "")
                    rep = d.get("rep_index")
                    ts = d.get("generation_timestamp", "") or ""
                    if not vid or not isinstance(rep, int):
                        continue
                    merged[(vid, rep, ts)] = {
                        "verdict": d.get("verdict", "pending"),
                        "confidence": float(d.get("confidence") or 0.0),
                        "evidence": d.get("evidence", ""),
                        "error": d.get("error", ""),
                        "question_template": d.get("question_template", ""),
                        "source": path.name,
                    }
        except OSError:
            continue
    return merged


_JUDGE_PASS2_FLAGS: Dict[Tuple[str, int, str], Dict[str, Any]] = _load_judge_pass2_flags()


_PASS2_VERDICT_EMOJI = {
    "side_matches": "✅",
    "side_mismatch": "↔️",
    "side_not_applicable": "—",
    "parse_failed": "⚠️",
    "error": "💥",
    "pending": "⏳",
}


def _sample_pass2_verdicts(sample: Dict) -> List[Dict[str, Any]]:
    """Return Pass-2 verdicts that touch any rep referenced by this sample.

    Pass 2 is per-(sample, rep) so we look up by the sample's
    generation_timestamp first (exact match), and only fall back to
    timestamp="" / any-timestamp when no exact match exists for that rep.
    """
    md = sample.get("metadata") or {}
    vid = md.get("video_id", "")
    ts = md.get("generation_timestamp", "") or ""
    if not vid:
        return []
    out: List[Dict[str, Any]] = []
    for rep in _sample_rep_indices(sample):
        # Exact match first.
        rec = _JUDGE_PASS2_FLAGS.get((vid, rep, ts))
        if rec is None:
            # Fall back to the empty-timestamp bucket if present.
            rec = _JUDGE_PASS2_FLAGS.get((vid, rep, ""))
        if rec is None:
            # Last resort: any verdict for this (video, rep) regardless of ts.
            for k, v in _JUDGE_PASS2_FLAGS.items():
                if k[0] == vid and k[1] == rep:
                    rec = v
                    break
        if rec:
            out.append({"rep_index": rep, **rec})
    return out


def _sample_rep_indices(sample: Dict) -> List[int]:
    """Return all rep indices a sample touches: from verification.rep_index,
    rep_comparison.rep_a/rep_b, verification.window_reps,
    verification.rep_indices, etc."""
    md = sample.get("metadata") or {}
    verif = md.get("verification") or {}
    rc = md.get("rep_comparison") or {}
    out: List[int] = []
    for k in ("rep_index",):
        v = verif.get(k)
        if isinstance(v, int):
            out.append(v)
    for k in ("rep_a", "rep_b"):
        v = rc.get(k)
        if isinstance(v, int):
            out.append(v)
    for src in (verif.get("window_reps"), verif.get("rep_indices"),
                md.get("window_reps"), md.get("rep_indices")):
        if isinstance(src, list):
            out.extend(int(x) for x in src if isinstance(x, int))
    # de-dup, preserve order
    seen: set = set()
    dedup: List[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def _sample_judge_verdicts(sample: Dict) -> List[Dict[str, Any]]:
    """Return all judge verdicts that touch any rep referenced by this sample."""
    vid = (sample.get("metadata") or {}).get("video_id", "")
    if not vid:
        return []
    out = []
    for rep in _sample_rep_indices(sample):
        rec = _JUDGE_FLAGS.get((vid, rep))
        if rec:
            out.append({"rep_index": rep, **rec})
    return out


_JUDGE_CATEGORY_EMOJI = {
    "ok": "✅",
    "patient_not_performing": "🚫",
    "multi_rep_in_one_rep": "🔁",
    "wrong_exercise": "❓",
    "pose_estimation_wrong_subject": "📍",
    "wrong_side_for_question": "↔️",
    "parse_failed": "⚠️",
    "error": "💥",
    "pending": "⏳",
}


def _rep_display_label(sample: Dict, disk_rep_index: int) -> str:
    """Translate a 0-indexed disk rep_index into a human-readable label
    that lines up with how the question text refers to it.

    Three cases:
      - Tier B comparative samples: question text uses positional labels
        "Rep 1" (= rep_a) and "Rep 2" (= rep_b). Display as
        ``"rep 2 (= question's Rep 1)"``.
      - Tier A samples: only one rep is referenced; the question doesn't
        use a numeric rep label. Display as ``"rep 2"`` (raw disk index).
      - Tier C/D and aggregate templates: window_reps / rep_indices, no
        single positional label. Display as raw ``"rep N"``.

    The disk index is ALWAYS 0-based (matches repetition_{N}/ on disk).
    """
    md = sample.get("metadata") or {}
    rc = md.get("rep_comparison") or {}
    rep_a = rc.get("rep_a")
    rep_b = rc.get("rep_b")
    if disk_rep_index == rep_a and isinstance(rep_a, int):
        return f"rep {disk_rep_index} (= question's Rep 1)"
    if disk_rep_index == rep_b and isinstance(rep_b, int):
        return f"rep {disk_rep_index} (= question's Rep 2)"
    return f"rep {disk_rep_index}"


def render_judge_panel(sample: Dict) -> str:
    """Build a Markdown block summarizing all judge verdicts (Pass 1 +
    Pass 2) that touch any rep this sample references. Designed for the
    'Judge Verdict (LLM)' accordion in the sample-detail panel.

    Labeling: rep numbers shown here are 0-indexed disk indices (matching
    repetition_{N}/ on disk and the metadata). For Tier B comparative
    samples we additionally annotate which one is "Rep 1" / "Rep 2" in
    the question text. See ``test_rep_index_contract.py`` for the pinned
    contract.
    """
    vid = (sample.get("metadata") or {}).get("video_id", "")
    rep_indices = _sample_rep_indices(sample)
    if not vid:
        return "*No video_id on sample*"
    if not _JUDGE_FLAGS and not _JUDGE_PASS2_FLAGS:
        return "*No judge_flags_*.jsonl loaded — see JUDGE_FLAGS_CANDIDATE_PATHS / JUDGE_PASS2_CANDIDATE_PATHS in app.py*"
    if not rep_indices:
        return f"*Sample references no rep indices — judge runs per (video, rep)*"

    pass2_records = {v["rep_index"]: v for v in _sample_pass2_verdicts(sample)}

    p1_lines: List[str] = []
    p2_lines: List[str] = []
    any_p1 = False
    any_p2 = False
    sources: set = set()

    def _detail_text(rec: Dict[str, Any]) -> str:
        """Return the most-informative one-liner for a record. Errors get
        their `error` field; normal verdicts get `evidence`."""
        ev = (rec.get("evidence") or "").strip()
        err = (rec.get("error") or "").strip()
        if err:
            # Show error first, evidence after if both exist.
            return f"⚠️ error: {err}" + (f"  •  evidence: {ev}" if ev else "")
        return ev or "_(no evidence)_"

    def _short_source(src: str) -> str:
        """Strip the `.jsonl` extension and any `judge_` prefix for a compact
        inline tag. e.g. 'judge_v14_widefix_v2.jsonl' → 'v14_widefix_v2'.
        This makes the per-verdict provenance scannable when multiple judge
        runs are merged (later files override earlier ones for the same
        (video, rep) key — see JUDGE_FLAGS_CANDIDATE_PATHS)."""
        s = src.rsplit("/", 1)[-1]
        if s.endswith(".jsonl"):
            s = s[:-len(".jsonl")]
        if s.startswith("judge_"):
            s = s[len("judge_"):]
        if s.startswith("flags_"):
            s = s[len("flags_"):]
        return s

    for rep in rep_indices:
        rep_label = _rep_display_label(sample, rep)
        # Pass 1
        rec1 = _JUDGE_FLAGS.get((vid, rep))
        if rec1:
            any_p1 = True
            cat = rec1.get("category", "?")
            conf = float(rec1.get("confidence", 0.0))
            emoji = _JUDGE_CATEGORY_EMOJI.get(cat, "•")
            src1 = rec1.get("source", "")
            src_tag = f"  •  _from_ `{_short_source(src1)}`" if src1 else ""
            p1_lines.append(
                f"- **{rep_label}**: {emoji} `{cat}` — conf **{conf:.2f}**{src_tag}\n"
                f"  > {_detail_text(rec1)}"
            )
            if src1:
                sources.add(src1)
        else:
            p1_lines.append(f"- **{rep_label}**: *(no verdict)*")

        # Pass 2
        rec2 = pass2_records.get(rep)
        if rec2:
            any_p2 = True
            v = rec2.get("verdict", "?")
            conf = float(rec2.get("confidence", 0.0))
            emoji = _PASS2_VERDICT_EMOJI.get(v, "•")
            src2 = rec2.get("source", "")
            src_tag = f"  •  _from_ `{_short_source(src2)}`" if src2 else ""
            p2_lines.append(
                f"- **{rep_label}**: {emoji} `{v}` — conf **{conf:.2f}**{src_tag}\n"
                f"  > {_detail_text(rec2)}"
            )
            if src2:
                sources.add(src2)
        else:
            p2_lines.append(f"- **{rep_label}**: *(no verdict)*")

    if not any_p1 and not any_p2:
        return f"*No judge verdict on any rep for `{vid}` (checked disk reps: {rep_indices})*"

    header = (
        f"**Video**: `{vid}`  •  **Disk reps referenced by this question**: "
        f"{rep_indices} (0-indexed; matches repetition_N/ on disk)"
    )
    body = []
    body.append("**Pass 1 — per-rep video-content audit:**")
    body.extend(p1_lines if any_p1 else ["*(no Pass-1 verdicts)*"])
    body.append("")
    body.append("**Pass 2 — side-mismatch check (uses the question text):**")
    body.extend(p2_lines if any_p2 else ["*(no Pass-2 verdicts)*"])
    footer = f"\n\n_Source(s): {', '.join(sorted(sources))}_" if sources else ""

    # Per-video aggregate — collapsed section listing every rep of this video
    # that has any judge verdict, beyond the reps the current sample touches.
    # Useful when the same video appears across multiple samples / multi-rep
    # templates and you want the full per-video picture without flipping
    # through samples.
    rep_set_in_sample = set(rep_indices)
    other_p1 = sorted(
        [(k[1], rec) for k, rec in _JUDGE_FLAGS.items()
         if k[0] == vid and k[1] not in rep_set_in_sample],
        key=lambda kv: kv[0],
    )
    # _JUDGE_PASS2_FLAGS keys are (video_id, rep_index, timestamp) — note 3-tuple.
    # Collapse to most-recent per rep so we don't double-count multiple snapshots.
    p2_by_rep: Dict[int, Dict[str, Any]] = {}
    for k, rec in _JUDGE_PASS2_FLAGS.items():
        if k[0] != vid:
            continue
        rep_i = k[1]
        if rep_i in rep_set_in_sample:
            continue
        # Last write wins (later entries override earlier ones — matches
        # the loader's merge-order convention).
        p2_by_rep[rep_i] = rec
    other_p2 = sorted(p2_by_rep.items(), key=lambda kv: kv[0])
    if other_p1 or other_p2:
        details_lines = []
        if other_p1:
            details_lines.append("**Pass 1 — other reps of this video:**\n")
            for r, rec in other_p1:
                cat = rec.get("category", "?")
                conf = float(rec.get("confidence", 0.0))
                emoji = _JUDGE_CATEGORY_EMOJI.get(cat, "•")
                src = rec.get("source", "")
                src_tag = f"  •  _from_ `{_short_source(src)}`" if src else ""
                details_lines.append(
                    f"- **rep {r}**: {emoji} `{cat}` — conf **{conf:.2f}**{src_tag}\n"
                    f"  > {_detail_text(rec)}"
                )
        if other_p2:
            if details_lines:
                details_lines.append("")
            details_lines.append("**Pass 2 — other reps of this video:**\n")
            for r, rec in other_p2:
                v = rec.get("verdict", "?")
                conf = float(rec.get("confidence", 0.0))
                emoji = _PASS2_VERDICT_EMOJI.get(v, "•")
                src = rec.get("source", "")
                src_tag = f"  •  _from_ `{_short_source(src)}`" if src else ""
                details_lines.append(
                    f"- **rep {r}**: {emoji} `{v}` — conf **{conf:.2f}**{src_tag}\n"
                    f"  > {_detail_text(rec)}"
                )
        n_other = len(other_p1) + len(other_p2)
        # HTML <details> renders as a collapsed accordion in Markdown.
        collapsed = (
            f"\n\n<details>\n"
            f"<summary>Other judged reps of this video "
            f"(<b>{n_other}</b> verdict{'s' if n_other != 1 else ''} on reps not in this sample)</summary>\n\n"
            + "\n".join(details_lines)
            + "\n\n</details>"
        )
    else:
        collapsed = ""

    return header + "\n\n" + "\n".join(body) + footer + collapsed


# Per-version annotation lineage. When loading annotations for the key on the
# left, each predecessor on the right is merged in first; entries from the
# current version override. Keys are JSONL stems (filename without .jsonl).
# Why: v13.1 inherits the same generation_timestamps as v13 for unchanged
# samples, so a "Has issues" flag the colleague made on v13 should still
# surface in the app when v13.1 is loaded.
ANNOTATION_LINEAGE: Dict[str, List[str]] = {
    "qa_all_exercises_v13_1": ["qa_all_exercises_v13"],
}


def _annotations_path_for_stem(stem: str) -> Path:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return ANNOTATIONS_DIR / f"{stem}_annotations.json"


def _load_annotations(jsonl_path: str) -> dict:
    """Load annotations for a JSONL, merging in any predecessor versions
    declared in ANNOTATION_LINEAGE. Current-version entries override
    predecessors when the same generation_timestamp key exists in both."""
    stem = Path(jsonl_path).stem
    merged: dict = {}
    for predecessor_stem in ANNOTATION_LINEAGE.get(stem, []):
        pre_path = _annotations_path_for_stem(predecessor_stem)
        if pre_path.exists():
            with open(pre_path) as f:
                merged.update(json.load(f))
    own_path = _annotations_path_for_stem(stem)
    if own_path.exists():
        with open(own_path) as f:
            merged.update(json.load(f))
    return merged


def _save_annotation(jsonl_path: str, key: str, entry: dict, all_annotations: dict) -> dict:
    all_annotations[key] = entry
    path = _annotations_path(jsonl_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_annotations, f, indent=2)
    return all_annotations


def _sample_key(parsed: dict) -> str:
    """Unique key from generation_timestamp (globally unique across all samples)."""
    return parsed["metadata"].get("generation_timestamp", "")


def _dataset_version(jsonl_path: str) -> str:
    return Path(jsonl_path).stem


def _annotation_counter_md(annotations: dict) -> str:
    if not annotations:
        return "*No annotations yet*"
    good = sum(1 for v in annotations.values() if v["rating"] == "Good quality")
    redundant = sum(1 for v in annotations.values() if v["rating"] == "Maybe redundant")
    issues = sum(1 for v in annotations.values() if v["rating"] == "Has issues")
    return f"**{len(annotations)}** annotations: {good} good, {redundant} redundant, {issues} issues"


def load_exercise_csv(csv_path: str) -> pd.DataFrame:
    if not Path(csv_path).exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path, dtype=str).fillna("")


def build_exercise_video_index(report_path: str) -> Dict[str, List[str]]:
    """Map exercise_code -> [video_ids] using zip_path prefix from report."""
    if not Path(report_path).exists():
        return {}
    with open(report_path) as f:
        report = json.load(f)
    index: Dict[str, List[str]] = {}
    for video_id, info in report.items():
        zip_path = info.get("zip_path", "")
        if zip_path:
            # zip_path: "vid_dir/EXERCISE_CODE_member_datetime_session.zip"
            zip_name = zip_path.split("/")[-1]
            exercise_code = zip_name.split("_")[0]
            index.setdefault(exercise_code, []).append(video_id)
    return index


# Exercises requiring BlazePose (hand/wrist/finger/plantar — no VitPose keypoints)
_BLAZEPOSE_EXERCISE_RE = re.compile(
    r"finger|thumb|hand\s*tendon|pinch|open\s*finger|wrist|nerve\s*glide.*wrist"
    r"|plantar\s*flexion|dorsiflexion",
    re.I,
)


def _resolve_kp_source_for_sample(sample: dict) -> str:
    """Return the keypoint/feature source for a SAMPLE: 'sam3dbody',
    'blazepose', or 'vitpose'.

    Sample-aware superset of `_resolve_kp_source` (which only sees a
    video_id). SAM-3D-Body drives the mesh / pose-class 3D features, while
    the joint-angle Tier A metrics use 2D VitPose keypoints. We detect
    SAM-3D-Body from the generator's own `computation_method` text
    (authoritative) and fall back to the per-video resolver for everything
    else. Backwards compatible: samples with no SAM-3D signal classify
    exactly as before (vitpose/blazepose), so existing datasets and the
    saved index buckets are unchanged.
    """
    meta = sample.get("metadata", {}) or {}
    v = meta.get("verification", {}) or {}

    # 0) If the generator already stamped the source into metadata, trust it
    #    (questions should always carry `metric_source`; see generator).
    stamped = meta.get("metric_source") or v.get("metric_source")
    if stamped in ("sam3dbody", "vitpose", "blazepose"):
        return stamped

    # 1) BlazePose wins by VIDEO source. Hand / wrist / finger / plantar
    #    exercises have NO VitPose keypoints — their 2D landmarks (and any
    #    derived metric, e.g. wrist ROM) come from BlazePose. This overrides
    #    the template→source maps below, which assume a VitPose/SAM-3D video.
    if _resolve_kp_source(meta.get("video_id", "")) == "blazepose":
        return "blazepose"

    tmpl = meta.get("question_template", "") or sample.get("template", "")
    # 2) Authoritative template→source maps: the pose3d feature extractor's
    #    primary input is SAM-3D-Body (`*_vitpose_3d_extras.json`, 3D
    #    keypoints / joint angles). A specific subset of Tier-A templates was
    #    deliberately re-implemented on 2D VitPose keypoints (the
    #    `keypoints_2d` functions in pose3d_features.py).
    if tmpl in _VITPOSE_2D_TEMPLATES:
        return "vitpose"
    if tmpl in _SAM3DBODY_TEMPLATES:
        return "sam3dbody"
    # 3) computation_method text as a secondary signal for templates not in
    #    either map (e.g. future generators).
    method = (meta.get("computation_method") or v.get("computation_method") or "")
    ml = method.lower()
    if "sam-3d" in ml or "sam3d" in ml or "sam 3d" in ml or "3d mesh" in ml or "3d_extras" in ml or "_meshes" in ml:
        return "sam3dbody"
    if "2d keypoint" in ml or "2d, " in ml or "from 2d" in ml:
        return "vitpose"
    # 4) Fall back to the per-video VitPose/BlazePose resolver (non-3D MCQA).
    return _resolve_kp_source(meta.get("video_id", ""))


# Tier-A 3D templates computed from 2D VitPose keypoints (the `keypoints_2d`
# functions in pose3d_features.py). Everything else 3D is SAM-3D-Body.
_VITPOSE_2D_TEMPLATES = {
    "tier_a_hip_extension_3d",
    "tier_a_hip_hyperextension_3d",
    "tier_a_prone_arm_lift_3d",
    "tier_a_prone_press_up_3d",
    "tier_a_shoulder_extension_3d",
    "tier_a_knee_pushup_3d",
    "tier_a_standing_row_3d",
    "tier_a_trunk_side_bend_3d",
    "tier_a_sidelying_abduction_3d",
    "tier_a_quad_stretch_3d",          # retired, kept for legacy datasets
    "tier_a_quad_stretch_depth_3d",    # uses _knee_flexion_rom (2D)
}

# 3D-feature templates computed from SAM-3D-Body (3D keypoints / joint
# angles / mesh): pose-class, motion plane, active side, trunk/neck/shoulder
# 3D angles, hip-hinge/abduction, limb extension, wrist ROM, plank, holds,
# and the generic joint-angle ROM/peak templates.
_SAM3DBODY_TEMPLATES = {
    "tier_a_pose_class",
    "tier_a_motion_plane",
    "tier_a_active_side_3d",
    "tier_a_trunk_lean_direction",
    "tier_a_trunk_rotation_direction",
    "tier_a_trunk_sagittal_direction",
    "tier_a_trunk_sagittal_rom",
    "tier_a_trunk_axial_yaw_rom",
    "tier_a_trunk_stability_hold_3d",
    "tier_a_plank_back_bend_3d",
    "tier_a_neck_rotation",
    "tier_a_neck_flexion_3d",
    "tier_a_shoulder_er_3d",
    "tier_a_shoulder_elevation_rom_3d",
    "tier_a_elbow_flexion_rom_3d",
    "tier_a_hip_hinge_rom_3d",
    "tier_a_hip_flexion_3d",
    "tier_a_hip_abduction_rom_3d",
    "tier_a_knee_flexion_3d",
    "tier_a_lower_body_depth_3d",
    "tier_a_limb_extension_arm_3d",
    "tier_a_limb_extension_leg_3d",
    "tier_a_wrist_rom_3d",
}

# All 3D-feature MCQA templates (pose3d pipeline). Used to suppress the
# legacy "Joint · Keypoints" header for these — they carry no meaningful
# verification.joint and print their own authoritative `Source` row.
_3D_FEATURE_TEMPLATE_NAMES = _VITPOSE_2D_TEMPLATES | _SAM3DBODY_TEMPLATES | {
    "tier_b_axial_vs_lean_comparison",
    "tier_b_compensatory_3d",
    "tier_c_coordination_3d",
}


def _resolve_kp_source(video_id: str) -> str:
    """Return 'blazepose' or 'vitpose' for this video.

    Reads the per-exercise `keypoint_source` flag from the metadata CSV,
    matching what qa_generator._get_source_for_exercise does. Falls back
    to a name-based regex when the CSV column is missing or blank — this
    is the legacy behavior for older metadata files without an explicit
    `keypoint_source` column.
    """
    exercise_code = video_id.split("_")[0] if "_" in video_id else ""
    if exercise_code and not EXERCISE_DF.empty:
        match = EXERCISE_DF[EXERCISE_DF["exercise_code"] == exercise_code]
        if not match.empty:
            ks = str(match.iloc[0].get("keypoint_source", "") or "").strip().lower()
            if ks == "blazepose":
                return "blazepose"
            if ks == "vitpose":
                return "vitpose"
            # Fallback: regex on exercise name (legacy behavior).
            name = match.iloc[0].get("exercise_name", "")
            if _BLAZEPOSE_EXERCISE_RE.search(name):
                return "blazepose"
    return "vitpose"


def _resolve_vitpose_swap(video_id: str) -> bool:
    """Return True when left/right keypoint labels should be swapped at load time.

    Reads the per-exercise `vitpose_swap` flag from the metadata CSV. This is the
    SAME lookup the generator uses (qa_generator._get_swap_lr_for_exercise) —
    keeps the app's metric computation in sync with generator-written metadata.
    """
    exercise_code = video_id.split("_")[0] if "_" in video_id else ""
    if exercise_code and not EXERCISE_DF.empty:
        match = EXERCISE_DF[EXERCISE_DF["exercise_code"] == exercise_code]
        if not match.empty:
            flag = str(match.iloc[0].get("vitpose_swap", "") or "").strip().lower()
            return flag == "yes"
    return False


# ---------------------------------------------------------------------------
# Sidecar per-video data: the app reads pre-computed metrics, phases, fps,
# per-rep info from a JSONL sidecar keyed by video_id. This keeps what the
# app displays strictly consistent with what the generator saw (same swap_lr,
# same source). Live recomputation is only used as a fallback when the
# sidecar is absent (backwards-compat with older JSONL files).
# ---------------------------------------------------------------------------
_VIDEO_METRICS_SIDECAR: Dict[str, dict] = {}


def _sidecar_get(video_id: str, key: str, default=None):
    """Lookup a top-level key in the sidecar entry for a video."""
    entry = _VIDEO_METRICS_SIDECAR.get(video_id)
    if entry is None:
        return default
    return entry.get(key, default)


def _sidecar_rep_info(video_id: str, rep_index: int) -> Optional[dict]:
    """Return the per-rep info dict from sidecar.reps[rep_index], if present."""
    reps = _sidecar_get(video_id, "reps") or {}
    return reps.get(str(rep_index))


def _load_video_metrics_sidecar(jsonl_path: str) -> None:
    """Populate _VIDEO_METRICS_SIDECAR from <jsonl_stem>.per_video_metrics.jsonl."""
    global _VIDEO_METRICS_SIDECAR
    _VIDEO_METRICS_SIDECAR = {}
    p = Path(jsonl_path)
    sidecar = p.with_name(p.stem + ".per_video_metrics.jsonl")
    if not sidecar.is_file():
        logger.info(f"No per-video metrics sidecar at {sidecar}")
        return
    with sidecar.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            vid = row.get("video_id")
            if vid:
                _VIDEO_METRICS_SIDECAR[vid] = row
    logger.info(f"Loaded {len(_VIDEO_METRICS_SIDECAR)} per-video metrics entries from {sidecar}")


class _SidecarVideoMetrics:
    """VideoMetrics-shaped wrapper around a sidecar dict.

    Exposes the same attributes the app's render_metrics_table uses:
      video_id, exercise_code, num_reps, per_rep (joint -> [RepMetrics-like]),
      mean_rom_by_joint, cov_by_joint, rom_trend_by_joint, kp_source.
    """
    class _Rep:
        __slots__ = (
            "rep_index", "joint_name", "rom_degrees", "peak_angle_degrees",
            "min_angle_degrees", "peak_frame", "min_frame", "mean_angle_degrees",
            "num_frames", "start_angle_degrees", "end_angle_degrees",
            "concentric_velocity", "eccentric_velocity", "velocity_ratio",
            "hold_frames", "hold_time_seconds", "fps", "movement_label",
        )

        def __init__(self, d: dict) -> None:
            for slot in self.__slots__:
                setattr(self, slot, d.get(slot, 0 if slot not in ("joint_name", "movement_label") else ""))

    def __init__(self, d: dict) -> None:
        self.video_id = d.get("video_id", "")
        self.exercise_id = d.get("exercise_id")
        self.exercise_code = d.get("exercise_code")
        self.num_reps = d.get("num_reps", 0)
        self.mean_rom_by_joint = d.get("mean_rom_by_joint") or {}
        self.cov_by_joint = d.get("cov_by_joint") or {}
        self.rom_trend_by_joint = d.get("rom_trend_by_joint") or {}
        self.kp_source = d.get("kp_source", "vitpose")
        self.swap_lr = d.get("swap_lr", False)
        self.per_rep: Dict[str, list] = {
            joint: [self._Rep(r) for r in reps]
            for joint, reps in (d.get("per_rep") or {}).items()
        }


def get_video_metrics_cached(video_id: str):
    """Return per-video metrics, preferring the sidecar; fall back to live compute.

    The sidecar (written by generation/write_per_video_metrics.py) is the
    canonical source — same swap_lr and kp_source as the generator used.
    Live recomputation is a backwards-compat fallback when the sidecar is
    unavailable (e.g. loading an old JSONL).
    """
    if not hasattr(get_video_metrics_cached, "_cache"):
        get_video_metrics_cached._cache = {}
    cache = get_video_metrics_cached._cache
    if video_id in cache:
        return cache[video_id]

    # Preferred path: sidecar lookup (no computation)
    if video_id in _VIDEO_METRICS_SIDECAR:
        result = _SidecarVideoMetrics(_VIDEO_METRICS_SIDECAR[video_id])
        cache[video_id] = result
        return result

    # Fallback: live recomputation (only when sidecar missing)
    from metric_calculator import compute_video_metrics
    from utils.data_loader import load_video

    video_dir = resolve_video_dir(video_id)
    if not video_dir.exists():
        cache[video_id] = None
        return None

    source = _resolve_kp_source(video_id)
    swap_lr = _resolve_vitpose_swap(video_id)
    try:
        try:
            video_data = load_video(str(video_dir), source=source, swap_lr=swap_lr)
        except TypeError:
            # Local data_loader doesn't know about swap_lr. Fall back to the
            # no-swap call. This only matters for the legacy live-compute
            # path — the sidecar (preferred) was produced with the correct
            # swap already baked in.
            video_data = load_video(str(video_dir), source=source)
    except Exception as e:
        logger.warning(f"load_video failed for {video_id}: {e}")
        cache[video_id] = None
        return None
    if video_data.num_reps == 0:
        cache[video_id] = None
        return None

    try:
        result = compute_video_metrics(video_data)
        result.kp_source = source  # attach source for display
    except Exception as e:
        logger.warning(f"compute_video_metrics failed for {video_id}: {e}")
        cache[video_id] = None
        return None
    cache[video_id] = result
    return result


def get_video_phases_cached(video_id: str) -> Dict[int, List[str]]:
    """Return per-frame phase labels from the metrics sidecar (preferred),
    falling back to live events.json reads when the sidecar is absent.

    Returns {rep_index: [phase_label_per_frame]} using state codes:
    R=concentric, B=final, b=eccentric. First frame='initial', last='return'.
    """
    if not hasattr(get_video_phases_cached, "_cache"):
        get_video_phases_cached._cache = {}
    cache = get_video_phases_cached._cache
    if video_id in cache:
        return cache[video_id]

    # Preferred: read from the sidecar (data, not computation).
    side = _VIDEO_METRICS_SIDECAR.get(video_id)
    if side and "phases" in side:
        phases = {int(k): v for k, v in side["phases"].items()}
        cache[video_id] = phases
        return phases

    # Fallback: live read (only when sidecar is missing).
    from metric_calculator import load_rep_phases

    video_dir = resolve_video_dir(video_id)
    reps_dir = video_dir / "repetitions"
    phases: Dict[int, List[str]] = {}
    if reps_dir.exists():
        for rep_dir in sorted(reps_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            events_file = rep_dir / "events.json"
            try:
                idx = int(rep_dir.name.split("_")[-1])
            except (ValueError, IndexError):
                continue
            phase_labels = load_rep_phases(str(events_file))
            if phase_labels:
                phases[idx] = phase_labels

    cache[video_id] = phases
    return phases


def render_metrics_table(metrics, highlight_joint: str = "", phases: Optional[Dict[int, List[str]]] = None, highlight_rep: Optional[int] = None, highlight_reps: Optional[Dict[int, str]] = None, tier_d_side_means: Optional[Dict[str, tuple]] = None, priority_joints: Optional[List[str]] = None, active_side_per_rep: Optional[Dict] = None, restrict_to_reps: Optional[List[int]] = None) -> str:
    """Render VideoMetrics as markdown with per-rep table, velocity, and phase summary."""
    from metric_calculator import format_phase_summary

    # Build mapping: filesystem rep_index → sequential display number (1, 2, 3...)
    # Use only rep indices present in metrics (excludes trimmed first/last reps).
    metric_rep_indices: set = set()
    for reps in metrics.per_rep.values():
        for m in reps:
            metric_rep_indices.add(m.rep_index)
    display_num = {idx: n for n, idx in enumerate(sorted(metric_rep_indices), 1)}

    # Optional filter: restrict the displayed rows to the rep window the
    # question actually uses. Keeps display_num computed against ALL reps so
    # the rep numbering you see matches the canonical "Rep 1, 2, ..." in the
    # rest of the app — we just hide non-question rows.
    restrict_set: Optional[set] = None
    if restrict_to_reps:
        restrict_set = set(int(r) for r in restrict_to_reps if r is not None)
    # Filter phases to only show reps present in metrics
    if phases:
        phases = {k: v for k, v in phases.items() if k in metric_rep_indices}

    kp_source = getattr(metrics, "kp_source", "vitpose")
    source_label = "BlazePose" if kp_source == "blazepose" else "VitPose"
    lines = [f"### Kinematic Metrics ({metrics.num_reps} reps) — *Keypoints: {source_label}*\n"]

    if highlight_reps:
        # Multi-rep case (tier_b_rom_comparison): show both reps with labels
        parts = []
        for fs_rep, label in sorted(highlight_reps.items()):
            if fs_rep in display_num:
                parts.append(f"filesystem rep {fs_rep} = display Rep {display_num[fs_rep]} ({label})")
            else:
                parts.append(f"filesystem rep {fs_rep} ({label}) — trimmed from metrics ⚠️")
        lines.append(f"*⭐ Compared reps: {' | '.join(parts)}*\n")
    elif highlight_rep is not None:
        if highlight_rep in display_num:
            lines.append(f"*⭐ Question rep: filesystem rep {highlight_rep} = display Rep {display_num[highlight_rep]}*\n")
        else:
            lines.append(f"*⚠️ Question rep {highlight_rep} was trimmed from metrics (edge rep — not shown below)*\n")

    # Phase summary per rep (shown once at the top)
    if phases:
        lines.append("#### Exercise Phases\n")
        for rep_idx in sorted(phases.keys()):
            if rep_idx not in display_num:
                continue
            summary = format_phase_summary(phases[rep_idx])
            if summary:
                lines.append(f"**Rep {display_num[rep_idx]}**: {summary}\n")
        lines.append("")

    joint_order = sorted(metrics.per_rep.keys())
    # Priority joints go first (in order), then remaining sorted joints
    _priority = list(priority_joints) if priority_joints else ([highlight_joint] if highlight_joint else [])
    for pj in reversed(_priority):
        if pj in joint_order:
            joint_order.remove(pj)
            joint_order.insert(0, pj)
    for joint in joint_order:
        reps = metrics.per_rep[joint]
        if not reps:
            continue

        prefix = ">> " if joint == highlight_joint else ""
        lines.append(f"{prefix}#### {joint.replace('_', ' ').title()}\n")

        # Determine if this joint has a bilateral counterpart for active-side display
        _joint_side = "left" if joint.startswith("left_") else ("right" if joint.startswith("right_") else None)
        _opp_joint = None
        if _joint_side:
            _opp_side = "right" if _joint_side == "left" else "left"
            _opp_joint = _opp_side + "_" + joint[len(_joint_side) + 1:]
            if _opp_joint not in metrics.per_rep:
                _opp_joint = None
        _show_active = active_side_per_rep is not None and _opp_joint is not None

        if _show_active:
            lines.append("| Rep | \u2713 | Active | ROM | Start | Peak (frame) | Min (frame) | End | Mean | FPS | \u03c9 conc | \u03c9 ecc | Ratio | Hold |")
            lines.append("|-----|---|--------|-----|-------|-------------|------------|-----|------|-----|--------|-------|-------|------|")
        else:
            lines.append("| Rep | \u2713 | ROM | Start | Peak (frame) | Min (frame) | End | Mean | FPS | \u03c9 conc | \u03c9 ecc | Ratio | Hold |")
            lines.append("|-----|---|-----|-------|-------------|------------|-----|------|-----|--------|-------|-------|------|")

        for m in sorted(reps, key=lambda r: r.rep_index):
            # Show ALL reps; mark the ones used by the question with a star
            # via the dedicated \u2713 column. (Previously this loop skipped reps
            # outside `restrict_set` \u2014 that hid context.)
            in_question = (restrict_set is None) or (m.rep_index in restrict_set)
            # Velocity ratio warning marker (-1.0 = not computable)
            if m.velocity_ratio < 0:
                ratio_str = "N/A"
            else:
                ratio_str = f"{m.velocity_ratio:.1f}"
                if m.velocity_ratio > 2.0:
                    ratio_str += " \u26a0"
            if highlight_reps and m.rep_index in highlight_reps:
                rep_marker = f" \u2b50{highlight_reps[m.rep_index]}"
            elif highlight_rep is not None and m.rep_index == highlight_rep:
                rep_marker = " \u2b50"
            elif in_question and restrict_set is not None:
                # Multi-rep question with no specific highlight \u2014 star every
                # rep in the analysis window.
                rep_marker = " \u2b50"
            else:
                rep_marker = ""

            hold_str = f"{m.hold_frames}f" + (f" ({m.hold_time_seconds:.1f}s)" if getattr(m, "hold_time_seconds", 0) > 0 else "")
            fps_str = f"{getattr(m, 'fps', 0):.1f}" if getattr(m, "fps", 0) > 0 else "—"
            _lbl = getattr(m, "movement_label", "")
            correct_cell = "✓" if _lbl == "correct" else ("✗" if _lbl == "incorrect" else "—")

            if _show_active:
                # ACTIVE SIDE IS READ-ONLY FROM METADATA. The app does NOT compute
                # it — metadata.active_side_per_rep is the single source of truth
                # (written by qa_generator._detect_active_side_per_rep). If the
                # map lacks an entry, or the value is "rest"/"ambiguous", we
                # render em-dash rather than guessing.
                _rep_key = str(m.rep_index)
                if _rep_key in active_side_per_rep:
                    _active_side = active_side_per_rep[_rep_key]
                elif "all" in active_side_per_rep:
                    _active_side = active_side_per_rep["all"]
                else:
                    _active_side = None
                if _active_side in ("left", "right"):
                    _active_marker = "\u2713" if _active_side == _joint_side else "\u2717"
                elif _active_side == "ambiguous":
                    _active_marker = "?"
                else:
                    _active_marker = "\u2014"  # em dash (—)
                lines.append(
                    f"| {display_num.get(m.rep_index, m.rep_index)}{rep_marker} | {correct_cell} | {_active_marker} "
                    f"| {m.rom_degrees:.1f}\u00b0 "
                    f"| {m.start_angle_degrees:.1f}\u00b0 "
                    f"| {m.peak_angle_degrees:.1f}\u00b0 (f{m.peak_frame}) "
                    f"| {m.min_angle_degrees:.1f}\u00b0 (f{m.min_frame}) "
                    f"| {m.end_angle_degrees:.1f}\u00b0 "
                    f"| {m.mean_angle_degrees:.1f}\u00b0 "
                    f"| {fps_str} "
                    f"| {m.concentric_velocity:.1f}\u00b0/f "
                    f"| {m.eccentric_velocity:.1f}\u00b0/f "
                    f"| {ratio_str} "
                    f"| {hold_str} |"
                )
            else:
                lines.append(
                    f"| {display_num.get(m.rep_index, m.rep_index)}{rep_marker} | {correct_cell} "
                    f"| {m.rom_degrees:.1f}\u00b0 "
                    f"| {m.start_angle_degrees:.1f}\u00b0 "
                    f"| {m.peak_angle_degrees:.1f}\u00b0 (f{m.peak_frame}) "
                    f"| {m.min_angle_degrees:.1f}\u00b0 (f{m.min_frame}) "
                    f"| {m.end_angle_degrees:.1f}\u00b0 "
                    f"| {m.mean_angle_degrees:.1f}\u00b0 "
                    f"| {fps_str} "
                    f"| {m.concentric_velocity:.1f}\u00b0/f "
                    f"| {m.eccentric_velocity:.1f}\u00b0/f "
                    f"| {ratio_str} "
                    f"| {hold_str} |"
                )

        cov = metrics.cov_by_joint.get(joint) or 0
        trend = metrics.rom_trend_by_joint.get(joint, "stable")
        mean_rom = metrics.mean_rom_by_joint.get(joint, 0)
        total_reps = len(reps)

        # Compute mean for the analyzed rep subset (highlighted reps)
        analyzed_mean: Optional[float] = None
        analyzed_n = 0
        _analyzed_indices: set = set()
        if highlight_reps:
            _analyzed_indices = set(highlight_reps.keys())
        elif highlight_rep is not None:
            _analyzed_indices = {highlight_rep}
        if _analyzed_indices:
            subset = [rm.rom_degrees for rm in reps if rm.rep_index in _analyzed_indices]
            if subset:
                analyzed_mean = sum(subset) / len(subset)
                analyzed_n = len(subset)

        if tier_d_side_means and joint in tier_d_side_means:
            left_mean, right_mean = tier_d_side_means[joint]
            footer = (
                f"\n**Mean ROM ({total_reps} reps)**: {mean_rom:.1f}\u00b0 | "
                f"**Left-side reps**: {left_mean:.1f}\u00b0 | **Right-side reps**: {right_mean:.1f}\u00b0 | "
                f"**CoV**: {cov:.1f}% | **Trend**: {trend}"
            )
        else:
            footer = f"\n**Mean ROM ({total_reps} reps)**: {mean_rom:.1f}\u00b0 | **CoV**: {cov:.1f}% | **Trend**: {trend}"

        if analyzed_mean is not None and analyzed_n != total_reps:
            footer += f" | **Mean ({analyzed_n} analyzed \u2605)**: {analyzed_mean:.1f}\u00b0"

        lines.append(footer + "\n")

    return "\n".join(lines)


# Global indexes (built at startup)
EXERCISE_DF = pd.DataFrame()
EXERCISE_VIDEO_INDEX: Dict[str, List[str]] = {}
PROCESSING_REPORT: Dict = {}
VIDEO_DIR_MAP: Dict[str, str] = {}  # report video_id -> actual directory name


def _build_video_dir_map(report: Dict) -> Dict[str, str]:
    """Map report video_id (3-part) to filesystem dir name (4-part with exercise code prefix).

    Report key: 'member_datetime_session'
    Dir name:   'exerciseCode_member_datetime_session' (from zip_path prefix)
    """
    mapping = {}
    for video_id, info in report.items():
        zip_path = info.get("zip_path", "")
        if zip_path:
            zip_name = zip_path.split("/")[-1].replace(".zip", "")
            mapping[video_id] = zip_name
    return mapping


def resolve_video_dir(video_id: str, video_path: str = "") -> Path:
    """Resolve a video_id to its actual filesystem directory.

    Checks in order:
    1. metadata.video_path from the JSONL row (most reliable — set at generation time)
    2. CONFIG["data_dir"] / video_id  (10k/all)
    3. VIDEO_DIR_MAP lookup (3-part report IDs)
    4. Additional known data roots (10k_2, 10k_complement)
    """
    # 1. Use video_path from metadata (provided or from module-level cache)
    vp = video_path or _VIDEO_PATH_CACHE.get(video_id, "")
    if vp:
        p = Path(vp)
        if p.exists():
            return p

    data_dir = Path(CONFIG["data_dir"])
    # 2. Direct match under primary data dir
    direct = data_dir / video_id
    if direct.exists():
        return direct

    # 3. Mapped name (3-part report IDs -> 4-part dir)
    mapped = VIDEO_DIR_MAP.get(video_id, "")
    if mapped:
        mapped_dir = data_dir / mapped
        if mapped_dir.exists():
            return mapped_dir

    # 4. Additional data roots (videos not in 10k/all) — built-ins plus any roots
    #    registered at runtime via CONFIG["extra_data_roots"] (e.g. mesh_viewer
    #    registers the _processed root for each _KNOWN_VIDEOS session so its bg
    #    frames resolve for the overlay video).
    alt_roots = [
        Path("/mnt/data/shared/vlm/data/10k_2"),
        Path("/mnt/data/shared/vlm/data/10k_complement"),
    ]
    alt_roots += [Path(r) for r in (CONFIG.get("extra_data_roots") or [])]
    for alt_root in alt_roots:
        candidate = alt_root / video_id
        if candidate.exists():
            return candidate

    return direct  # fallback (will be non-existent)


def get_video_labels(video_id: str) -> Dict[int, str]:
    """Get movement labels for each rep of a video. Returns {rep_index: label}.

    Prefers sidecar data; falls back to reading movement_label.txt from disk.
    """
    # Preferred: sidecar
    reps_sc = _sidecar_get(video_id, "reps") or {}
    if reps_sc:
        labels = {}
        for k, info in reps_sc.items():
            try:
                idx = int(k)
            except (ValueError, TypeError):
                continue
            lbl = (info.get("movement_label") or "").strip().lower()
            if lbl:
                labels[idx] = lbl
        if labels:
            return labels

    # Fallback: disk
    video_dir = resolve_video_dir(video_id)
    reps_dir = video_dir / "repetitions"
    labels: Dict[int, str] = {}
    if reps_dir.exists():
        for rep_dir in reps_dir.iterdir():
            if rep_dir.is_dir() and rep_dir.name.startswith("repetition_"):
                try:
                    rep_idx = int(rep_dir.name.split("_")[1])
                except (ValueError, IndexError):
                    continue
                label_file = rep_dir / "movement_label.txt"
                if label_file.exists():
                    labels[rep_idx] = label_file.read_text().strip().lower()
    return labels


def get_video_label_summary(video_id: str) -> str:
    """Summarize movement labels for a video: 'correct', 'incorrect', 'mixed', or 'unknown'."""
    labels = get_video_labels(video_id)
    if not labels:
        return "unknown"
    unique = set(labels.values()) - {"unknown"}
    if unique == {"correct"}:
        return "correct"
    elif unique == {"incorrect"}:
        return "incorrect"
    elif "incorrect" in unique and "correct" in unique:
        return "mixed"
    return "unknown"


def init_global_indexes():
    global EXERCISE_DF, EXERCISE_VIDEO_INDEX, PROCESSING_REPORT, VIDEO_DIR_MAP
    csv_path = CONFIG["exercise_csv"]
    if Path(csv_path).exists():
        EXERCISE_DF = load_exercise_csv(csv_path)
        print(f"  Loaded {len(EXERCISE_DF)} exercises from {csv_path}")

    report_path = CONFIG["processing_report"]
    if Path(report_path).exists():
        EXERCISE_VIDEO_INDEX = build_exercise_video_index(report_path)
        with open(report_path) as f:
            PROCESSING_REPORT = json.load(f)
        VIDEO_DIR_MAP = _build_video_dir_map(PROCESSING_REPORT)
        print(f"  Loaded {len(PROCESSING_REPORT)} videos from processing report")
        print(f"  Mapped {len(EXERCISE_VIDEO_INDEX)} exercise codes to videos")
        print(f"  Built dir map for {len(VIDEO_DIR_MAP)} videos")


# ---------------------------------------------------------------------------
# 5. Skeleton overlay
# ---------------------------------------------------------------------------

# COCO-17 skeleton connections with left/right coloring (BGR)
SKELETON_CONNECTIONS = [
    # (joint_a_idx, joint_b_idx, color_bgr)
    (5, 7, (255, 0, 0)),    # left_shoulder -> left_elbow (blue=left)
    (7, 9, (255, 0, 0)),    # left_elbow -> left_wrist
    (6, 8, (0, 0, 255)),    # right_shoulder -> right_elbow (red=right)
    (8, 10, (0, 0, 255)),   # right_elbow -> right_wrist
    (5, 6, (0, 255, 255)),  # left_shoulder -> right_shoulder (yellow=center)
    (5, 11, (255, 0, 0)),   # left_shoulder -> left_hip
    (6, 12, (0, 0, 255)),   # right_shoulder -> right_hip
    (11, 12, (0, 255, 255)),  # left_hip -> right_hip
    (11, 13, (255, 0, 0)),  # left_hip -> left_knee
    (13, 15, (255, 0, 0)),  # left_knee -> left_ankle
    (12, 14, (0, 0, 255)),  # right_hip -> right_knee
    (14, 16, (0, 0, 255)),  # right_knee -> right_ankle
]

COCO_17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# BlazePose-extra connections (name-based, drawn when keypoints are present)
BLAZEPOSE_EXTRA_CONNECTIONS = [
    # Hand landmarks: wrist -> index/pinky/thumb
    ("left_wrist", "left_index", (255, 128, 0)),     # orange-blue
    ("left_wrist", "left_pinky", (255, 128, 0)),
    ("left_wrist", "left_thumb", (255, 128, 0)),
    ("right_wrist", "right_index", (0, 128, 255)),    # orange-red
    ("right_wrist", "right_pinky", (0, 128, 255)),
    ("right_wrist", "right_thumb", (0, 128, 255)),
    # Foot landmarks: ankle -> heel -> foot_index
    ("left_ankle", "left_heel", (255, 128, 0)),
    ("left_heel", "left_foot_index", (255, 128, 0)),
    ("left_ankle", "left_foot_index", (255, 128, 0)),
    ("right_ankle", "right_heel", (0, 128, 255)),
    ("right_heel", "right_foot_index", (0, 128, 255)),
    ("right_ankle", "right_foot_index", (0, 128, 255)),
]

JOINT_COLORS = {}
for name in COCO_17_NAMES:
    if "left" in name:
        JOINT_COLORS[name] = (255, 0, 0)   # blue (BGR)
    elif "right" in name:
        JOINT_COLORS[name] = (0, 0, 255)   # red
    else:
        JOINT_COLORS[name] = (0, 255, 255)  # yellow


def draw_skeleton_on_frame(
    frame_bgr,
    keypoints: Dict[str, Tuple[float, float, float]],
    is_normalized: bool = True,
    conf_threshold: float = 0.3,
):
    """Draw skeleton overlay on a BGR frame (in-place).
    Draws COCO-17 base skeleton + BlazePose hand/foot landmarks when present."""
    import cv2
    h, w = frame_bgr.shape[:2]

    # Resolve all keypoints to pixel coords (name -> (px, py))
    all_pts: Dict[str, Tuple[int, int]] = {}
    for name, (x, y, conf) in keypoints.items():
        if conf < conf_threshold:
            continue
        px = int(x * w) if is_normalized else int(x)
        py = int(y * h) if is_normalized else int(y)
        all_pts[name] = (px, py)

    # Build indexed pts for COCO-17 connections
    pts = {}
    for i, name in enumerate(COCO_17_NAMES):
        if name in all_pts:
            pts[i] = all_pts[name]

    # Draw COCO-17 connections
    for a, b, color in SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame_bgr, pts[a], pts[b], color, 2, cv2.LINE_AA)

    # Draw BlazePose extra connections (hand/foot)
    for name_a, name_b, color in BLAZEPOSE_EXTRA_CONNECTIONS:
        if name_a in all_pts and name_b in all_pts:
            cv2.line(frame_bgr, all_pts[name_a], all_pts[name_b], color, 2, cv2.LINE_AA)

    # Draw COCO-17 joints
    for i, (px, py) in pts.items():
        name = COCO_17_NAMES[i]
        color = JOINT_COLORS.get(name, (0, 255, 255))
        cv2.circle(frame_bgr, (px, py), 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame_bgr, (px, py), 4, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw BlazePose extra joints (hand/foot landmarks)
    _EXTRA_NAMES = {"left_index", "right_index", "left_pinky", "right_pinky",
                    "left_thumb", "right_thumb", "left_heel", "right_heel",
                    "left_foot_index", "right_foot_index", "left_palm_wrist", "right_palm_wrist"}
    for name in _EXTRA_NAMES:
        if name in all_pts:
            px, py = all_pts[name]
            color = (255, 128, 0) if "left" in name else (0, 128, 255)
            cv2.circle(frame_bgr, (px, py), 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)


# The two canonical per-video frame source directories:
# - cropped_images/             → cropped frames without skeleton
# - cropped_images_keypoints/   → cropped frames with skeleton overlay pre-rendered
# Both are flat directories (one .webp per frame timestamp). Rep membership is
# resolved via cropped_repetitions/repetition_N (used as a filename-to-rep index
# only — never for actual display paths).


def _frame_source_dir(video_id: str, skeleton: bool) -> Path:
    """Return the canonical flat source dir for display frames.

    Prefers cropped_images[_keypoints]/. If the cropped variant is absent on
    disk (happens for some upstream-incomplete videos), falls back to the
    uncropped images/ dir. For skeleton mode, falls back to non-skeleton
    sources too — better to show the video without skeleton overlay than to
    show nothing at all.
    """
    vd = resolve_video_dir(video_id)
    if skeleton:
        skel = vd / "cropped_images_keypoints"
        if skel.is_dir():
            return skel
        # Fall through to non-skeleton sources rather than returning a
        # nonexistent path. Caller can detect this (skeleton requested,
        # got non-skeleton path) but won't render an empty player.
    cropped = vd / "cropped_images"
    if cropped.is_dir():
        return cropped
    uncropped = vd / "images"
    if uncropped.is_dir():
        return uncropped
    return cropped


def _rep_filenames(video_id: str, rep_index: int) -> List[str]:
    """Return the ordered list of frame FILENAMES that belong to a given rep.

    Uses cropped_repetitions/repetition_N as the index. Does NOT return paths —
    callers must resolve the filename into cropped_images/ or
    cropped_images_keypoints/ depending on whether skeleton overlay is desired.
    """
    vd = resolve_video_dir(video_id)
    for subdir in ("cropped_repetitions", "repetitions"):
        d = vd / subdir / f"repetition_{rep_index}"
        if d.exists():
            names = sorted(p.name for p in d.glob("*.webp"))
            if names:
                return names
    return []


def _resolve_filenames_to_source(video_id: str, filenames: List[str], skeleton: bool) -> List[str]:
    """Map a list of frame filenames to full paths in the canonical source dir.

    Silently drops filenames that don't exist in the chosen source dir — this
    can happen because cropped_images_keypoints/ is slightly sparser than
    cropped_images/ (frames where no keypoints were detected are omitted).
    """
    src = _frame_source_dir(video_id, skeleton=skeleton)
    if not src.exists():
        return []
    available = {p.name: str(p) for p in src.glob("*.webp")}
    return [available[fn] for fn in filenames if fn in available]


# ---------------------------------------------------------------------------
# 5b. SAM-3D-Body mesh / 3D overlay resolution  (consumed by mesh_viewer.py)
#
# These were referenced by mesh_viewer.py before the API was implemented; added
# 2026-06-30 during the video-sft-vlm archive so the mesh viewer (port 7871)
# actually runs. SAM-3D outputs (`*_3d_extras.json`, `*_3d_meshes.npz`) are
# co-located INSIDE the rep directory (see feedback_sam3d_extras_colocate).
# The heavy rasterization lives in app_mesh_render.render_rep_frames.
# ---------------------------------------------------------------------------

# Modes that need a SAM-3D render (anything other than raw / 2D skeleton).
_3D_MODES = {
    "mesh",
    "3d_skel",
    "mesh_kp_combined",
    "side_by_side_raw_mesh",
    "side_by_side_mesh_skel",
}

# Cache of rendered overlay mp4s + their RenderStats, keyed by (video, rep, mode).
_MESH_VIDEO_CACHE: Dict[Tuple[str, int, str], str] = {}
_MESH_STATS_CACHE: Dict[Tuple[str, int, str], Dict[str, Any]] = {}


def _sam3d_searched_dirs(video_id: str, rep_index: int) -> List[Path]:
    """Candidate directories where THIS video_id's SAM-3D extras would live.

    Two shapes are supported, both scoped to `video_id` (never a global flat list —
    a shared audit dir must NOT match an unrelated video):
      1. Co-located rep dirs:  <video_dir>/{cropped_repetitions_3d,..}/repetition_N
      2. A per-video root registered in CONFIG["sam3d_video_roots"][video_id]
         (mesh_viewer's _KNOWN_VIDEOS 3rd element). That root is checked as a flat
         dir (extras directly inside, e.g. pose_class_audit/<x>/) AND as a
         <root>/<video_id>/cropped_repetitions_3d/repetition_N rep tree.
    """
    vd = resolve_video_dir(video_id)
    # SAM-3D extras co-locate in a sibling `*_3d` rep tree (run_sam3d_vo.py writes
    # cropped_repetitions_3d/repetition_N/), and historically also directly inside
    # the rep dir. Check both.
    cands = [vd / sub / f"repetition_{rep_index}"
             for sub in ("cropped_repetitions_3d", "repetitions_3d",
                         "cropped_repetitions", "repetitions")]
    # Only THIS video's registered root — keyed by video_id, so a shared audit dir
    # (e.g. pose_class_audit/supine, which belongs to exactly one video) can never
    # be returned for a different video.
    root = (CONFIG.get("sam3d_video_roots") or {}).get(video_id)
    if root:
        rp = Path(root)
        cands.append(rp)                 # flat audit dir (extras directly inside)
        cands.append(rp / video_id)      # <root>/<video_id> flat
        for sub in ("cropped_repetitions_3d", "repetitions_3d"):
            cands.append(rp / video_id / sub / f"repetition_{rep_index}")
    return cands


def _sam3d_output_dir(video_id: str, rep_index: int) -> Optional[Path]:
    """Return the directory that holds SAM-3D outputs for (video, rep).

    A directory qualifies only if it contains a `*_3d_extras.json` sidecar.
    Returns None when no SAM-3D data has been generated for this rep.
    """
    for d in _sam3d_searched_dirs(video_id, rep_index):
        if d.exists() and any(d.glob("*_3d_extras.json")):
            return d
    return None


def _rep_frame_paths_for_3d(video_id: str, rep_index: int,
                            sam3d_dir: Path) -> List[str]:
    """Background frame paths for a 3D overlay, in the extras' frame order.

    SAM-3D ran on the per-rep crop (`cropped_repetitions/repetition_N/` or
    `repetitions/repetition_N/`) — the only frames whose dims match the extras'
    image_width/height, so the only ones the mesh projection aligns to. We order
    them by the extras' `file_name` sequence so frame i of the render lines up
    with frame i of the 3D data.
    """
    vd = resolve_video_dir(video_id)
    rep_dir = None
    for sub in ("cropped_repetitions", "repetitions"):
        d = vd / sub / f"repetition_{rep_index}"
        if d.is_dir() and any(d.glob("*.webp")):
            rep_dir = d
            break
    if rep_dir is None:
        return []
    by_name = {p.name: str(p) for p in rep_dir.glob("*.webp")}
    # Order by the extras' file_name list when available; else sorted on disk.
    ordered_names: List[str] = []
    extras_hits = list(sam3d_dir.glob("*_3d_extras.json"))
    if extras_hits:
        try:
            ex = json.loads(extras_hits[0].read_text())
            ordered_names = [f.get("file_name", "") for f in ex.get("frames", [])]
        except Exception:
            ordered_names = []
    if not ordered_names:
        ordered_names = sorted(by_name.keys())
    return [by_name[n] for n in ordered_names if n in by_name]


def _resolve_video_for_mode(video_id: str, rep_index: int, mode: str) -> Optional[str]:
    """Return a playable mp4 path for the requested overlay mode.

    - `raw` / `skeleton`: delegate to the normal rep-video pipeline (2D).
    - 3D modes: render frames via app_mesh_render and encode to mp4 (cached).
    Returns None if the inputs can't be resolved.
    """
    if mode not in _3D_MODES:
        # Non-3D modes use the standard rep video (skeleton overlay handled there).
        return get_or_create_video(video_id, rep_index=rep_index,
                                    skeleton=(mode == "skeleton"))

    key = (video_id, rep_index, mode)
    if key in _MESH_VIDEO_CACHE and os.path.exists(_MESH_VIDEO_CACHE[key]):
        return _MESH_VIDEO_CACHE[key]

    sam3d_dir = _sam3d_output_dir(video_id, rep_index)
    if sam3d_dir is None:
        return None

    # The mesh's pred_cam_t / focal_length are aligned to the EXACT crop SAM-3D
    # ran on — the per-rep `cropped_repetitions/repetition_N/` (or `repetitions/`)
    # frames, whose dims equal the extras' image_width/height. Compositing onto
    # cropped_images/ (a DIFFERENT, larger crop) floats the mesh off the body
    # (the head-above-head offset). So resolve bg frames from the rep dir the
    # extras reference, matched by the extras' file_name order. See the prior
    # investigation: PLAN_VIDEO_RENDERER §"background is always the cropped frame".
    bg_paths = _rep_frame_paths_for_3d(video_id, rep_index, sam3d_dir)
    if not bg_paths:
        return None

    from app_mesh_render import render_rep_frames  # local import (heavy deps)

    cache_root = Path(CONFIG["video_cache_dir"]) / "mesh_render"
    out_dir = cache_root / video_id / f"rep{rep_index}_{mode}"
    png_paths, stats = render_rep_frames(bg_paths, sam3d_dir, mode, out_dir)
    if not png_paths:
        return None

    fps = get_video_fps(video_id) or CONFIG.get("default_fps", 8.0)
    out_mp4 = str(cache_root / f"{video_id}_rep{rep_index}_{mode}.mp4")
    hflip = bool(_VIDEO_FLIP_CACHE.get(video_id, False))
    encode_video(png_paths, fps=fps, output_path=out_mp4, hflip=hflip)

    _MESH_VIDEO_CACHE[key] = out_mp4
    _MESH_STATS_CACHE[key] = {
        "n_frames": stats.n_frames,
        "n_missing_3d": stats.n_missing_3d,
        "n_off_canvas": stats.n_off_canvas,
        "mode": stats.mode,
    }
    return out_mp4


def _load_mesh_render_stats(video_id: str, rep_index: int, mode: str) -> Optional[Dict[str, Any]]:
    """Return cached RenderStats kwargs for a rendered (video, rep, mode), if any."""
    return _MESH_STATS_CACHE.get((video_id, rep_index, mode))


def get_skeleton_frames(
    video_id: str, rep_index: int = 1, source: str = "vitpose",
) -> Optional[str]:
    """Return the directory containing skeleton-annotated frames for a rep.

    Uses the canonical cropped_images_keypoints/ source (flat dir of pre-rendered
    frames with skeleton overlay). Returns the source-dir path; callers filter
    by rep via _rep_filenames when needed.

    The `source` and `rep_index` args are retained for API compatibility but no
    longer trigger runtime skeleton drawing — we always use the pre-rendered
    cropped_images_keypoints/ frames so the overlay is produced by the same
    upstream pipeline that generated the keypoints.
    """
    src = _frame_source_dir(video_id, skeleton=True)
    return str(src) if src.exists() else None


# ---------------------------------------------------------------------------
# 6. Video encoding
# ---------------------------------------------------------------------------

def _annotate_frames_with_rep_labels(
    paths: List[str],
    boundaries: List[Tuple[int, int]],
    video_id: str = "",
) -> List[str]:
    """Burn 'Rep N' labels onto frames so the video player shows current rep.

    Uses ThreadPoolExecutor for parallel I/O (~2-3s for 300 frames).
    Returns list of annotated frame paths in a temp directory.
    """
    import cv2
    import tempfile as _tf
    from concurrent.futures import ThreadPoolExecutor

    tmp_dir = _tf.mkdtemp(prefix="rep_label_")
    vid_labels = get_video_labels(video_id) if video_id else {}

    # Pre-compute per-frame metadata: (label, frame_in_rep, total_in_rep)
    frame_meta: List[Tuple[str, int, int]] = []
    for rep_idx, n_frames in boundaries:
        label = f"Rep {rep_idx}"
        movement = vid_labels.get(rep_idx, "")
        if movement and movement != "unknown":
            label += f" ({movement})"
        for f in range(n_frames):
            frame_meta.append((label, f + 1, n_frames))

    def process_frame(args):
        i, path = args
        frame = cv2.imread(path)
        if frame is None:
            return path
        h, w = frame.shape[:2]
        label, frame_in_rep, total_in_rep = frame_meta[i] if i < len(frame_meta) else ("", 0, 0)

        bar_h = max(28, int(h * 0.06))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.0, w / 500))
        thickness = max(1, int(font_scale * 2))
        cv2.putText(frame, label, (8, bar_h - 8), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        counter = f"{frame_in_rep}/{total_in_rep}"
        (tw, _), _ = cv2.getTextSize(counter, font, font_scale * 0.8, thickness)
        cv2.putText(frame, counter, (w - tw - 8, bar_h - 8), font, font_scale * 0.8,
                    (200, 200, 200), thickness, cv2.LINE_AA)

        out_path = os.path.join(tmp_dir, f"{i:05d}.webp")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_WEBP_QUALITY, 50])
        return out_path

    with ThreadPoolExecutor(max_workers=8) as pool:
        annotated = list(pool.map(process_frame, enumerate(paths)))

    return annotated


def get_video_fps(video_id: str, fallback_dir: str = "",
                  prefer_dir: bool = False) -> float:
    """Get fps for a video.

    When `prefer_dir=True` and a `fallback_dir` is supplied, we read the
    on-disk fps.txt first — this is the right behavior for per-rep video
    rendering, where each rep has its own fps that differs slightly from
    the session-wide average stored in the JSONL metadata cache.

    Default (prefer_dir=False): metadata cache → fps.txt → default.
    """
    if prefer_dir and fallback_dir:
        return read_fps(fallback_dir)
    if video_id in _VIDEO_FPS_CACHE:
        return _VIDEO_FPS_CACHE[video_id]
    if fallback_dir:
        return read_fps(fallback_dir)
    return CONFIG["default_fps"]


def encode_video(image_paths: List[str], fps: float, output_path: str, hflip: bool = False) -> str:
    """Encode WebP frames to H.264 MP4 for browser playback."""
    import shutil
    import subprocess
    import tempfile

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Find ffmpeg binary (imageio-ffmpeg bundles one)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    if shutil.which(ffmpeg) is None:
        raise RuntimeError(
            "Video encoding requires ffmpeg. Install the system 'ffmpeg' binary "
            "or add the Python package 'imageio-ffmpeg' to the active environment."
        )

    # Write frame list to temp file for ffmpeg concat
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for path in image_paths:
            duration = 1.0 / fps
            f.write(f"file '{path}'\nduration {duration:.6f}\n")
        # Repeat last frame WITH duration to avoid cut
        f.write(f"file '{image_paths[-1]}'\nduration {duration:.6f}\n")
        f.write(f"file '{image_paths[-1]}'\n")
        list_file = f.name

    try:
        vf_filters = ["hflip"] if hflip else []
        vf_filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        # `-r fps` forces the output container's frame rate to match the
        # source. Without this, libx264 defaults to 25 fps regardless of the
        # per-frame `duration` PTS hints in the concat list — players that
        # honour container fps (most browsers) play slow-fps clips too fast.
        cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
            "-vf", ",".join(vf_filters),
            "-r", f"{fps:.4f}",
            "-c:v", "libx264", "-preset", "fast", "-tune", "zerolatency",
            "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to encode video: {stderr}") from exc
    finally:
        os.unlink(list_file)

    return output_path


def get_rep_frame_paths(video_id: str, rep_index: int = 1,
                        use_cropped: bool = False) -> List[str]:
    """Get sorted .webp frame paths for a repetition.

    Always resolves paths from the canonical flat source dir
    (cropped_images/ or cropped_images_keypoints/) via _rep_filenames. The
    legacy `use_cropped` flag is retained for API compatibility but is now a
    no-op — cropped frames are the only supported source.
    """
    del use_cropped  # always use canonical cropped source
    filenames = _rep_filenames(video_id, rep_index)
    return _resolve_filenames_to_source(video_id, filenames, skeleton=False)


def _iter_rep_indices(video_id: str) -> List[int]:
    """Return the ordered list of rep indices present for a video."""
    vd = resolve_video_dir(video_id)
    for subdir in ("cropped_repetitions", "repetitions"):
        d = vd / subdir
        if d.exists():
            return sorted(
                int(p.name.split("_")[1])
                for p in d.iterdir()
                if p.is_dir() and p.name.startswith("repetition_") and p.name.split("_")[1].isdigit()
            )
    return []


def get_allreps_frame_paths(
    video_id: str, use_cropped: bool = False, skeleton: bool = False,
    trim_edge_reps: bool = True,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Get frame paths across ALL reps concatenated in order.

    Args:
        trim_edge_reps: If True (default), exclude first and last rep to match
            metric computation trimming. Requires >=3 reps; if fewer, no trimming.
        skeleton: If True, resolve paths in cropped_images_keypoints/; else cropped_images/.
        use_cropped: No-op (retained for API compatibility).

    Returns (all_paths, boundaries) where boundaries = [(display_number, num_frames), ...]
    Display numbers are sequential 1-based (1, 2, 3...) after trimming.
    """
    del use_cropped
    rep_indices = _iter_rep_indices(video_id)
    raw_entries: List[Tuple[int, List[str]]] = []
    for idx in rep_indices:
        fns = _rep_filenames(video_id, idx)
        paths = _resolve_filenames_to_source(video_id, fns, skeleton=skeleton)
        if paths:
            raw_entries.append((idx, paths))

    if trim_edge_reps and len(raw_entries) >= 3:
        raw_entries = raw_entries[1:-1]

    all_paths: List[str] = []
    boundaries: List[Tuple[int, int]] = []
    for display_num, (_, frames) in enumerate(raw_entries, 1):
        boundaries.append((display_num, len(frames)))
        all_paths.extend(frames)

    return all_paths, boundaries


def get_selected_reps_frame_paths(
    video_id: str, rep_indices: List[int], use_cropped: bool = False, skeleton: bool = False,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Get frame paths for specific rep indices only.

    Returns (all_paths, boundaries) where boundaries = [(display_number, num_frames), ...]
    with sequential 1-based display numbers.
    """
    del use_cropped
    all_paths: List[str] = []
    boundaries: List[Tuple[int, int]] = []
    for display_num, rep_idx in enumerate(sorted(rep_indices), 1):
        fns = _rep_filenames(video_id, rep_idx)
        frames = _resolve_filenames_to_source(video_id, fns, skeleton=skeleton)
        if frames:
            boundaries.append((display_num, len(frames)))
            all_paths.extend(frames)
    return all_paths, boundaries


def _sample_with_captions(paths, max_frames, total_frames):
    """Sample paths evenly and return (path, caption) tuples with frame numbers."""
    if not paths:
        return []
    if len(paths) <= max_frames:
        indices = list(range(len(paths)))
    else:
        step = len(paths) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
    return [(paths[i], f"{i + 1}/{total_frames}") for i in indices]


def _sample_allreps_with_captions(paths, boundaries, max_frames):
    """Sample frames across all reps with rep-context captions like 'R1 3/20'."""
    if not paths:
        return []
    total = len(paths)
    if total <= max_frames:
        indices = list(range(total))
    else:
        step = total / max_frames
        indices = [int(i * step) for i in range(max_frames)]

    result = []
    for idx in indices:
        cumulative = 0
        caption = f"{idx + 1}/{total}"
        for rep_idx, n_frames in boundaries:
            if idx < cumulative + n_frames:
                frame_in_rep = idx - cumulative + 1
                caption = f"R{rep_idx} {frame_in_rep}/{n_frames}"
                break
            cumulative += n_frames
        result.append((paths[idx], caption))
    return result


def get_gallery_frames(video_id: str, rep_index: int = 1,
                       max_frames: int = 0, skeleton: bool = False,
                       use_cropped: bool = False, all_reps: bool = False):
    """Get evenly-sampled frame paths for a gallery. Returns (path, caption) tuples."""
    if max_frames == 0:
        max_frames = CONFIG["max_gallery_frames"]

    if all_reps:
        paths, boundaries = get_allreps_frame_paths(video_id, use_cropped=use_cropped, skeleton=skeleton)
        # Show ~3 frames per rep so every rep is represented
        allreps_max = max(max_frames, len(boundaries) * 3)
        return _sample_allreps_with_captions(paths, boundaries, allreps_max)

    paths = _get_rep_frames(video_id, rep_index, skeleton=skeleton)
    if not paths:
        return []
    total = len(paths)
    if total <= max_frames:
        return [(p, f"{i + 1}/{total}") for i, p in enumerate(paths)]
    step = total / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    return [(paths[i], f"{i + 1}/{total}") for i in indices]


def get_or_create_video(video_id: str, rep_index: int = 1,
                        skeleton: bool = False, use_cropped: bool = False,
                        all_reps: bool = False) -> Optional[str]:
    """Get cached video or encode from frames."""
    cache_dir = CONFIG["video_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    hflip = _VIDEO_FLIP_CACHE.get(video_id, False)
    flip_suffix = "_flip" if hflip else ""

    if all_reps:
        suffix = "_allreps_v4" + ("_skel" if skeleton else "") + ("_crop" if use_cropped else "") + flip_suffix + "_fpsfix"
        cache_key = f"{video_id}{suffix}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cache_path = os.path.join(cache_dir, f"{cache_hash}.mp4")
        if os.path.exists(cache_path):
            return cache_path
        paths, boundaries = get_allreps_frame_paths(video_id, use_cropped=use_cropped, skeleton=skeleton)
        if not paths:
            return None
        fps = get_video_fps(video_id, str(Path(paths[0]).parent))
        annotated = _annotate_frames_with_rep_labels(paths, boundaries, video_id=video_id)
        try:
            return encode_video(annotated, fps, cache_path, hflip=hflip)
        finally:
            import shutil
            shutil.rmtree(Path(annotated[0]).parent, ignore_errors=True)

    suffix = "_skel" if skeleton else ("_crop" if use_cropped else "")
    cache_key = f"{video_id}_rep{rep_index}{suffix}{flip_suffix}_fpsfix"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"{cache_hash}.mp4")

    if os.path.exists(cache_path):
        return cache_path

    paths = _get_rep_frames(video_id, rep_index, skeleton=skeleton)
    if not paths:
        return None
    # Per-rep fps: prefer the rep's own fps.txt over the session-wide cache.
    fps = get_video_fps(
        video_id,
        str(resolve_video_dir(video_id) / "repetitions" / f"repetition_{rep_index}"),
        prefer_dir=True,
    )
    return encode_video(paths, fps, cache_path, hflip=hflip)


def _get_rep_frames(video_id: str, rep_index: int, skeleton: bool = False) -> List[str]:
    """Get sorted .webp frames for a single rep, with skeleton support."""
    filenames = _rep_filenames(video_id, rep_index)
    return _resolve_filenames_to_source(video_id, filenames, skeleton=skeleton)


def get_or_create_2rep_video(
    video_id: str, rep_a: int, rep_b: int, skeleton: bool = False,
) -> Optional[str]:
    """Build a video showing exactly 2 reps, labeled 'Rep 1' and 'Rep 2'."""
    cache_dir = CONFIG["video_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    hflip = _VIDEO_FLIP_CACHE.get(video_id, False)
    flip_suffix = "_flip" if hflip else ""
    suffix = f"_2rep_{rep_a}_{rep_b}" + ("_skel" if skeleton else "") + flip_suffix + "_fpsfix"
    cache_key = f"{video_id}{suffix}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"{cache_hash}.mp4")
    if os.path.exists(cache_path):
        return cache_path

    frames_a = _get_rep_frames(video_id, rep_a, skeleton=skeleton)
    frames_b = _get_rep_frames(video_id, rep_b, skeleton=skeleton)
    if not frames_a or not frames_b:
        return None

    all_paths = frames_a + frames_b
    # Label as "Rep 1" and "Rep 2" (display labels, not filesystem indices)
    boundaries = [(1, len(frames_a)), (2, len(frames_b))]

    # Use rep_a's per-rep fps (the two reps' fps differ by <1% in practice)
    fps = get_video_fps(
        video_id,
        str(resolve_video_dir(video_id) / "repetitions" / f"repetition_{rep_a}"),
        prefer_dir=True,
    )
    annotated = _annotate_frames_with_rep_labels(all_paths, boundaries, video_id=video_id)
    try:
        return encode_video(annotated, fps, cache_path, hflip=hflip)
    finally:
        import shutil
        shutil.rmtree(Path(annotated[0]).parent, ignore_errors=True)


def get_2rep_gallery_frames(
    video_id: str, rep_a: int, rep_b: int,
    skeleton: bool = False, max_frames: int = 0,
) -> List[Tuple[str, str]]:
    """Get gallery frames for exactly 2 reps, labeled 'Rep 1' and 'Rep 2'."""
    if max_frames == 0:
        max_frames = CONFIG["max_gallery_frames"]

    frames_a = _get_rep_frames(video_id, rep_a, skeleton=skeleton)
    frames_b = _get_rep_frames(video_id, rep_b, skeleton=skeleton)
    if not frames_a and not frames_b:
        return []

    all_paths = frames_a + frames_b
    boundaries = [(1, len(frames_a)), (2, len(frames_b))]
    allreps_max = max(max_frames, len(boundaries) * 3)
    return _sample_allreps_with_captions(all_paths, boundaries, allreps_max)


def get_gallery_from_frames(
    frame_paths: List[str], max_frames: int = 0,
) -> List[Tuple[str, str]]:
    """Gallery from an explicit frame list (production video_frames field)."""
    if not frame_paths:
        return []
    if max_frames == 0:
        max_frames = CONFIG["max_gallery_frames"]
    total = len(frame_paths)
    if total <= max_frames:
        return [(p, f"{i + 1}/{total}") for i, p in enumerate(frame_paths)]
    step = total / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    return [(frame_paths[i], f"{i + 1}/{total}") for i in indices]


def get_or_create_video_from_frames(
    frame_paths: List[str], fps: float, cache_key: str, hflip: bool = False,
) -> Optional[str]:
    """Encode an mp4 from an explicit frame list. cache_key must be unique per question."""
    if not frame_paths:
        return None
    cache_dir = CONFIG["video_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    # Include flip in cache key so flipped/unflipped don't collide
    full_key = cache_key + ("_flip" if hflip else "")
    cache_hash = hashlib.md5(full_key.encode()).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"{cache_hash}.mp4")
    if os.path.exists(cache_path):
        return cache_path
    return encode_video(frame_paths, fps or CONFIG["default_fps"], cache_path, hflip=hflip)


def get_video_metadata(video_id: str, rep_index: int = 1,
                       question_reps: Optional[List[str]] = None) -> str:
    """Build video metadata markdown for the accordion.

    question_reps: list of rep folder names relevant to this question
    (e.g. ["repetition_1", "repetition_3"]).  When provided, replaces the
    generic "Current Rep" line with a "Question Reps" block.
    """
    video_dir = resolve_video_dir(video_id)
    rep_dir = video_dir / "repetitions" / f"repetition_{rep_index}"

    lines = [f"**Video Directory**: `{video_dir}`"]

    # Build set of rep names used by the question (if any) so we can flag them.
    _question_rep_set = set(question_reps or [])

    # Prefer sidecar-provided rep list; fallback to filesystem scan.
    sc_reps = _sidecar_get(video_id, "reps") or {}
    if sc_reps:
        # Sort numerically by rep_index
        ordered = sorted(sc_reps.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 9_999)
        lines.append(f"**Repetitions**: {len(ordered)}")
        for k, info in ordered:
            rep_name = f"repetition_{k}"
            full = video_dir / "repetitions" / rep_name
            n_frames = info.get("num_frames", 0)
            marker = " ⭐ _in use_" if rep_name in _question_rep_set else ""
            lines.append(f"- `{full}` ({n_frames} frames){marker}")
    elif video_dir.exists():
        # Fallback: filesystem (only when sidecar absent)
        reps_dir = video_dir / "repetitions"
        if reps_dir.exists():
            rep_dirs = sorted(
                (d.name for d in reps_dir.iterdir()
                 if d.is_dir() and d.name.startswith("repetition_")),
                key=lambda n: int(n.split("_")[1]) if n.split("_")[1].isdigit() else 9_999,
            )
            lines.append(f"**Repetitions**: {len(rep_dirs)}")
            for rd in rep_dirs:
                full = reps_dir / rd
                n_frames = len(list(full.glob("*.webp")))
                marker = " ⭐ _in use_" if rd in _question_rep_set else ""
                lines.append(f"- `{full}` ({n_frames} frames){marker}")

    if question_reps and len(question_reps) > 1:
        # Multi-rep question: show which reps are relevant
        reps_dir = video_dir / "repetitions"
        fps = read_fps_for_video(video_id, str(rep_dir) if rep_dir.exists() else "")
        rep_frame_counts = {}
        for rep_name in question_reps:
            try:
                _idx = int(rep_name.split("_")[-1])
            except (ValueError, IndexError):
                _idx = None
            sc_info = _sidecar_rep_info(video_id, _idx) if _idx is not None else None
            if sc_info:
                rep_frame_counts[rep_name] = sc_info.get("num_frames", 0)
            else:
                rd = reps_dir / rep_name
                rep_frame_counts[rep_name] = len(list(rd.glob("*.webp"))) if rd.exists() else 0
        total_frames = sum(rep_frame_counts.values())
        rep_ids = [r.replace("repetition_", "") for r in question_reps]
        if len(question_reps) <= 6:
            detail = ", ".join(
                f"rep_{rid} ({rep_frame_counts[r]}f)"
                for rid, r in zip(rep_ids, question_reps)
            )
            lines.append(f"**Question Reps** ({len(question_reps)}): {detail} | total {total_frames} frames")
        else:
            lines.append(
                f"**Question Reps** ({len(question_reps)}): "
                f"rep_{rep_ids[0]}–rep_{rep_ids[-1]} | total {total_frames} frames"
            )
        lines.append(f"**FPS**: {fps}")
    else:
        # Single-rep view: prefer sidecar for FPS, frame count, movement label.
        fps = read_fps_for_video(video_id, str(rep_dir) if rep_dir.exists() else "")
        sc_info = _sidecar_rep_info(video_id, rep_index)
        if sc_info is not None:
            n_frames = sc_info.get("num_frames", 0)
            movement_label = sc_info.get("movement_label", "")
        elif rep_dir.exists():
            n_frames = len(list(rep_dir.glob("*.webp")))
            label_file = rep_dir / "movement_label.txt"
            movement_label = label_file.read_text().strip() if label_file.exists() else ""
        else:
            n_frames = 0
            movement_label = ""
        lines.append(f"**Current Rep**: repetition_{rep_index} | FPS: {fps} | Frames: {n_frames}")
        if movement_label:
            lines.append(f"**Movement Label**: {movement_label}")

    # Exercise code from zip_path in processing report
    report_info = PROCESSING_REPORT.get(video_id, {})
    zip_path = report_info.get("zip_path", "")
    if zip_path:
        exercise_code = zip_path.split("/")[-1].split("_")[0]
        lines.insert(1, f"**Exercise Code**: {exercise_code}")
    if report_info:
        lines.append(f"**Expected Reps**: {report_info.get('num_repetitions', '?')}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Renderers
# ---------------------------------------------------------------------------

_TEMPLATE_DISPLAY = {
    "tier_a_rom_single_rep":         "ROM Single Rep",
    "tier_a_peak_angle":             "Peak Angle",
    "tier_a_phase_identification":   "Phase Identification",
    "tier_a_temporal_grounding":     "Temporal Grounding",
    "tier_b_rom_comparison":         "ROM Comparison",
    "tier_b_peak_comparison":        "Peak Comparison",
    "tier_b_error_detection":        "Error Detection",
    "tier_b_correctness_criteria":   "Correctness Criteria",
    "tier_b_compensatory":           "Compensatory",
    "tier_c_kinematic_trend":        "Kinematic Trend",
    "tier_c_variability_analysis":   "Variability Analysis",
    "tier_c_trend_analysis":         "Trend Analysis",
    "tier_c_error_trend":            "Error Trend",
    "tier_d_side_comparison":        "Side ROM Comparison",
    "tier_d_asymmetry_detection":    "Asymmetry Detection",
    "tier_d_side_consistency":       "Side Consistency",
    "tier_d_peak_comparison":        "Side Peak Comparison",
    "tier_a_trunk_stability_hold_3d": "Plank Stability (3D)",
    "tier_a_plank_back_bend_3d":      "Plank Back Bend (3D)",
    "tier_a_shoulder_elevation_rom_3d": "Shoulder Elevation ROM (3D)",
    "tier_a_elbow_flexion_rom_3d":      "Elbow ROM (3D)",
    "tier_a_neck_flexion_3d":           "Neck Flexion/Side-bend (3D)",
    "tier_a_hip_abduction_rom_3d":      "Hip Abduction ROM (3D)",
    "tier_a_hip_hinge_rom_3d":          "Hip Hinge Depth (3D)",
    "tier_a_limb_extension_arm_3d":     "Limb Extension — Arm (3D)",
    "tier_a_limb_extension_leg_3d":     "Limb Extension — Leg (3D)",
    "tier_a_hip_extension_3d":          "Hip Extension at Peak (3D)",
    "tier_a_hip_hyperextension_3d":     "Hip Hyperextension — Leg Lift (3D)",
    "tier_a_prone_arm_lift_3d":         "Prone Arm Lift (3D)",
    "tier_a_prone_press_up_3d":         "Prone Press-Up (3D)",
    "tier_a_shoulder_extension_3d":     "Shoulder Extension (3D)",
    "tier_a_knee_pushup_3d":            "Knee Push-Up Depth (3D)",
    "tier_a_quad_stretch_3d":           "Quadriceps Stretch — Knee Flexion (3D)",
    "tier_a_quad_stretch_depth_3d":     "Quadriceps Stretch — Depth (3D)",
    "tier_a_trunk_side_bend_3d":        "Trunk Side-Bend ROM (3D)",
    "tier_a_sidelying_abduction_3d":    "Sidelying Hip Abduction ROM (3D)",
    "tier_a_standing_row_3d":           "Standing Row — Elbow Flexion (3D)",
}


def _escape_choice(text: str) -> str:
    """Escape markdown characters in choice text that cause unintended formatting.
    Tildes (~) cause strikethrough; underscores in joint definitions like
    left_knee → left_hip cause italics."""
    text = text.replace("~", "\\~")
    text = text.replace("_", "\\_")
    return text


def render_vo_block(parsed: Dict) -> str:
    """Render a vo3d Visual-Observations record: the geometry-derived answer block
    + per-question answers[] audit (chosen option, tier, defer reason, raw value,
    bin edges). Used when a record is one full [VISUAL OBSERVATIONS] block per rep
    rather than per-question MCQA."""
    answers = parsed.get("vo_answers") or []
    md = parsed.get("metadata", {})
    n_ans = sum(1 for a in answers if a.get("answered"))

    # Cross-check vs the human-grounded oracle GT (joined on session,rep).
    _load_oracle_obs()
    sess = md.get("video_id") or parsed.get("video_id", "")
    rep = md.get("rep_index")
    if rep is None:
        rep = parsed.get("metadata", {}).get("rep_index", -1)
    oracle_rec = _ORACLE_OBS_BY_REP.get((sess, int(rep) if rep is not None else -1), {})
    oracle = oracle_rec.get("by_num", {}) if oracle_rec else {}
    oracle_raw = oracle_rec.get("raw", "") if oracle_rec else ""
    has_oracle = bool(oracle)

    # Is this a vo3d GEOMETRY record (answers carry tier/reason/raw provenance) or
    # an SFT/oracle record (answers parsed straight from the [VISUAL OBSERVATIONS]
    # text, no geometry metadata)? Computed once and reused for every
    # geometry-only section below, so the panel can't half-render one mode.
    is_geometry = any(
        a.get("tier") or a.get("reason")
        or (a.get("verification") or {}).get("raw_measured_value") is not None
        for a in answers)
    # "rest deferred"/"Source" are geometry-isms; only show them for geometry rows.
    subhdr = (f"*{n_ans} of {len(answers)} answered (rest deferred). "
              f"Source: {md.get('_source_dataset','')}*\n") if is_geometry \
        else f"*SFT training row — {len(answers)} questions, gold response below*\n"
    lines = [f"### [VISUAL OBSERVATIONS] — {md.get('exercise_code','')} "
             f"{md.get('exercise_name','')}"]

    # Show the QUESTIONS that were asked (the "Visual Observations:" section of the
    # prompt, with options), so the panel isn't answer-only. Slice from the
    # questions marker to the <format> block; fall back to nothing if absent.
    qtext = parsed.get("question_text", "") or ""
    for marker in ("Visual Observations:", "Visual observations:"):
        if marker in qtext:
            qsec = qtext.split(marker, 1)[1]
            qsec = qsec.split("<format>", 1)[0].strip()
            if qsec:
                lines += ["**Questions asked**\n", qsec, "\n"]
            break

    lines += [subhdr,
              "**Correct response (gold target)**\n",
              "```", parsed.get("vo_block", "").strip(), "```\n"]

    # Our answers + verification — but ONLY for vo3d GEOMETRY records, where each
    # answer carries tier/raw/bin-edge provenance. SFT/oracle records (e.g. the
    # 1805 merged_reasoning sets) parse their answers straight from the
    # [VISUAL OBSERVATIONS] text block above and have no geometry metadata, so the
    # table would be all-empty/None and misleading. Skip it for SFT rows (see
    # is_geometry above) — the raw block already shows the answers as in the data.
    if is_geometry:
        lines += ["**Geometry answers (ours)**\n",
                  "| # | ans | option | tier | reason | raw | bin edges |",
                  "|---|---|---|---|---|---|---|"]
        for a in answers:
            v = a.get("verification", {}) or {}
            opt = (a.get("chosen_option") or "")
            lines.append(
                f"| {a.get('schema_question_index',0)+1} | "
                f"{'✅' if a.get('answered') else '·'} | {_md_cell(opt)} | "
                f"{a.get('tier','')} | {a.get('reason') or ''} | "
                f"{v.get('raw_measured_value')} | {v.get('bin_edges')} |")

    # Oracle cross-check — ONLY meaningful for vo3d GEOMETRY records, where a
    # separate 397B oracle is an independent reference. For SFT/merged rows the
    # gold response IS the resolved ground truth (the categorical+angle oracles
    # were already adjudicated via tiebreaker when the merged set was built), so
    # there is no higher authority to "cross-check" against — comparing back to a
    # single raw oracle is backwards and would falsely flag the angle schema.
    # Skip it entirely for those rows; the gold block above is the GT.
    if is_geometry:
        lines.append("")
        if has_oracle:
            lines += ["**Original GT (oracle) + cross-check**\n",
                      "| # | question | ours | oracle (GT) | match |",
                      "|---|---|---|---|---|"]
            for a in answers:
                qi = a.get("schema_question_index", 0)
                ours = (a.get("chosen_option") or "").strip()
                gt = (oracle.get(qi) or "").strip()
                if not a.get("answered"):
                    match = "— (deferred)"
                elif not gt:
                    match = "— (no GT)"
                elif ours == gt:
                    match = "✅"
                else:
                    match = "❌"
                q = (a.get("question_text") or "")
                lines.append(f"| {qi+1} | {_md_cell(q)} | {_md_cell(ours)} | "
                             f"{_md_cell(gt)} | {match} |")
            scored = [(a.get('chosen_option') or '').strip() == (oracle.get(a['schema_question_index']) or '').strip()
                      for a in answers if a.get('answered') and (oracle.get(a['schema_question_index']) or '').strip()]
            if scored:
                lines.append(f"\n*Agreement on answered+GT questions: "
                             f"{sum(scored)}/{len(scored)}*")
        else:
            lines.append("*No oracle GT found for this (session, rep) — "
                         "cross-check unavailable (rep not in 1105/1805 oracle-obs).*")

    # Raw HUMAN PT annotation — the actual ground truth the oracle was
    # conditioned on (analysis_of_movement, severities, ROM, feedback). Verbatim.
    _load_human_annot()
    ha = _HUMAN_ANNOT_BY_REP.get((sess, int(rep) if rep is not None else -1))
    if ha:
        lines.append("\n**Raw human PT annotation (ground truth)**\n")
        if ha.get("analysis_of_movement"):
            lines += ["*Movement analysis:*", "",
                      "> " + str(ha["analysis_of_movement"]).replace("\n", "\n> "), ""]
        if ha.get("severity_scores"):
            lines += ["*Severity scores:*", "```", str(ha["severity_scores"]).strip(), "```"]
        meta_bits = []
        for f in ("rom", "effectiveness", "injury_risk", "error_pattern", "agreement_level"):
            if ha.get(f) is not None:
                meta_bits.append(f"{f}={ha[f]}")
        if meta_bits:
            lines.append("*" + " · ".join(meta_bits) + "*")
        if ha.get("therapist_feedback"):
            lines += ["", "*Therapist feedback:* " + str(ha["therapist_feedback"])]
        # Everything else captured from the source row (nothing hidden).
        _shown = {"analysis_of_movement", "severity_scores", "rom", "effectiveness",
                  "injury_risk", "error_pattern", "agreement_level", "therapist_feedback"}
        extra = {k: v for k, v in ha.items()
                 if k not in _shown and v is not None and str(v).strip() != ""}
        if extra:
            lines += ["", "*Other annotation fields:*", "```"]
            for k in sorted(extra):
                lines.append(f"{k}: {str(extra[k]).strip()}")
            lines.append("```")
    else:
        lines.append("\n*No raw human PT annotation found for this (session, rep).*")

    # Raw oracle responses (the 397B oracle-obs block) — paired with the schema
    # question text for each numbered line (incl. questions we deferred).
    oracle_by_num = (oracle_rec or {}).get("by_num") or {}
    oracle_raw = (oracle_rec or {}).get("raw")
    if oracle_by_num or oracle_raw:
        lines.append("\n**Raw oracle responses (397B oracle-obs) — with questions**\n")
        qs = _schema_questions(md.get("exercise_code", ""))
        if oracle_by_num and qs:
            lines += ["| # | question | oracle answer |", "|---|---|---|"]
            for qi in sorted(oracle_by_num):
                qtext = qs[qi]["question"] if qi < len(qs) else "(question not in schema)"
                lines.append(f"| {qi+1} | {_md_cell(qtext)} | "
                             f"{_md_cell(oracle_by_num[qi])} |")
        else:
            # no schema match — show the verbatim block so nothing is lost
            lines += ["*(schema questions unavailable — showing the raw block)*",
                      "```", str(oracle_raw or "").strip(), "```"]
        # always also include the fully verbatim block for reference
        if oracle_raw:
            lines += ["\n*Verbatim oracle block:*", "```",
                      str(oracle_raw).strip(), "```"]
    else:
        lines.append("\n*No oracle responses found for this (session, rep) "
                     "— rep not in 1105/1805 oracle-obs.*")
    return "\n".join(lines)


def render_question(parsed: Dict) -> str:
    """Render question with choices, correct answer marked."""
    # vo3d Visual-Observations records have a full answer block, not MCQA choices.
    if parsed.get("vo_answers") is not None:
        return render_vo_block(parsed)
    tmpl_label = _TEMPLATE_DISPLAY.get(parsed.get("template", ""), parsed.get("template", ""))
    lines = [f"### Question ({tier_display(parsed['tier'])}) · *{tmpl_label}*\n"]

    # Ground-truth assistant answer (e.g. binary "Answer: yes/no" or any non-MCQA
    # target). parse_sample already extracts it into correct_letter with <think>
    # stripped. Show it so the panel isn't answer-less for non-MCQA datasets.
    answer = parsed.get("correct_letter", "")
    answer_md = ""
    if isinstance(answer, str) and answer and answer != "?" \
            and "[VISUAL OBSERVATIONS]" not in answer:
        # Strip a leading "Answer:" so binary targets don't read "Answer: Answer: yes".
        shown = answer.strip()
        shown = re.sub(r"^\s*answer\s*:\s*", "", shown, flags=re.IGNORECASE).strip()
        answer_md = f"\n**✅ Answer:** {_escape_choice(shown)}\n"

    # Extract just the question stem (before A))
    text = parsed["question_text"]
    # Split at choice options
    parts = text.split("\n\nA)")
    if len(parts) > 1:
        lines.append(_escape_choice(parts[0].strip()))
        lines.append("")
    else:
        lines.append(text)
        if answer_md:
            lines.append(answer_md)
        return "\n".join(lines)

    labels = string.ascii_uppercase
    choices = parsed["choices"]
    correct_idx = parsed["correct_index"]

    for i, choice in enumerate(choices):
        marker = "**-->**" if i == correct_idx else ""
        icon = " ✅" if i == correct_idx else ""
        lines.append(f"{marker} **{labels[i]})** {_escape_choice(choice)}{icon}\n")

    return "\n".join(lines)


def _md_cell(value: str) -> str:
    """Sanitise a string for safe insertion into a Markdown table cell.
    Replaces pipe characters (column breaks) and escapes underscores."""
    return str(value).replace("|", "\\|").replace("_", "\\_")


def render_verification(parsed: Dict) -> str:
    """Render tier-specific verification data."""
    # vo3d records carry per-answer verification inside answers[] (shown in the
    # question panel), not a single top-level block — skip the MCQA verification.
    if parsed.get("vo_answers") is not None:
        return ""
    v = parsed["verification"]
    tier = parsed["tier"]
    # Fallback chain covers geometry MCQA (joint), compensatory analysis
    # (compensatory_joint), and visual-obs Tier-0 samples (tier0_joint).
    joint = (
        v.get("joint")
        or v.get("compensatory_joint")
        or v.get("tier0_joint")
        or "?"
    )

    # Authoritative per-sample source (sam3dbody / vitpose / blazepose) —
    # same classifier the 3D box's `Source` row and the filter use, so the
    # header and the box never contradict each other.
    kp_source = _resolve_kp_source_for_sample(parsed)
    source_label = {
        "sam3dbody": "SAM-3D-Body",
        "blazepose": "BlazePose",
        "vitpose": "VitPose",
    }.get(kp_source, "VitPose")

    exercise_name = parsed.get("exercise_name", "") or v.get("exercise_name", "")
    lines = ["### Verification\n"]

    # Surface the human-salvage provenance banner first so it's the most
    # visible piece of context when a salvaged sample is opened. These
    # samples were rated `Good quality` by a human reviewer on a prior
    # dataset version but `needs_reverification` is typically True because
    # the surviving pool was dominated by an unattributed colleague's
    # ratings Sandra didn't re-rate under v13_1 standards (see
    # REPORT_SALVAGE_HUMAN.md §TL;DR).
    salvaged = (parsed.get("metadata") or {}).get("salvaged_human") or {}
    if salvaged:
        ver = salvaged.get("source_version", "?")
        rating = salvaged.get("annotation_rating", "?")
        author = salvaged.get("annotation_author", "?")
        note = (salvaged.get("annotation_note") or "").strip()
        needs_rev = salvaged.get("needs_reverification", False)
        badge = "🏷️ **Salvaged from prior version (human-rated)**"
        bits = [
            f"- Rated `{rating}` by **{author}** on **{ver}**",
        ]
        if note:
            bits.append(f"- Reviewer note: _{note}_")
        if needs_rev:
            bits.append(
                "- ⚠️ `needs_reverification`: rater is not `sgsilva`; was never "
                "re-rated under current standards. Treat as a salvage candidate, "
                "not a confirmed Good."
            )
        else:
            bits.append("- ✅ rater is `sgsilva`; treated as confirmed Good.")
        lines.append(badge + "\n" + "\n".join(bits) + "\n")

    if exercise_name:
        lines.append(f"**Exercise**: {exercise_name}\n")
    # 3D-feature templates don't carry a meaningful verification.joint (it
    # shows as "?") and they print an authoritative `Source` row inside the
    # metric box below — so skip the legacy "Joint · Keypoints" header for
    # them to avoid a contradictory / empty line.
    _is_3d_template = (parsed.get("template", "") or "").endswith("_3d") or \
        (parsed.get("template", "") in _3D_FEATURE_TEMPLATE_NAMES)
    if not _is_3d_template:
        if "TIER_D" in tier:
            right_joint = joint.replace("left_", "right_", 1) if joint.startswith("left_") else joint
            lines.append(f"**Joints**: {joint.replace('_', ' ')} / {right_joint.replace('_', ' ')} · *Keypoints: {source_label}*\n")
        else:
            lines.append(f"**Joint**: {joint.replace('_', ' ')} · *Keypoints: {source_label}*\n")

    # Visual-obs Tier-0 block: per-rep primitive value + the fixed bin
    # definitions (the value ranges we use to map primitive → MCQA option).
    if v.get("ground_truth_source") == "tier0_geometry":
        primitive = v.get("tier0_primitive", "")
        # Primitive-specific value field. Older amplitude samples used the
        # legacy field name; control samples carry tier0_velocity_ratio.
        primitive_value_field = {
            "peak_angle_degrees": "tier0_peak_angle_degrees",
            "max_angle_degrees": "tier0_max_angle_degrees",
            "velocity_ratio": "tier0_velocity_ratio",
        }.get(primitive)
        primitive_value = v.get(primitive_value_field) if primitive_value_field else None
        # Fallback for legacy samples missing tier0_primitive: assume amplitude.
        if primitive_value is None:
            primitive_value = v.get("tier0_peak_angle_degrees")
            if primitive_value is not None and not primitive:
                primitive = "peak_angle_degrees"
        # Human-readable label + unit for the metric row.
        primitive_label, primitive_unit = {
            "peak_angle_degrees": ("Peak Angle", "°"),
            "max_angle_degrees": ("Max Angle", "°"),
            "velocity_ratio": ("Velocity Ratio (ecc/conc)", ""),
        }.get(primitive, (primitive or "Primitive", ""))
        version = v.get("tier0_threshold_version", "")
        correct_letter = parsed.get("correct_letter", "")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        if primitive:
            lines.append(f"| Primitive | `{primitive}` |")
        if primitive_value is not None:
            lines.append(f"| {primitive_label} | {primitive_value:.2f}{primitive_unit} |")
        if version:
            lines.append(f"| Threshold version | {version} |")
        bin_defs = v.get("tier0_bin_definitions") or []
        if bin_defs:
            lines.append("")
            lines.append("**Bin definitions (fixed):**\n")
            for entry in bin_defs:
                marker = " ← this rep" if entry.startswith(f"{correct_letter}:") else ""
                lines.append(f"- {entry}{marker}")
        # Skip the rest of the tier-A/B/C/D branches for Tier-0 samples.
        return "\n".join(lines)

    # 3D-feature MCQA templates: render a template-specific box, then return.
    # These generators don't emit the rep_a/rep_b/rom_degrees fields the standard
    # renderer expects, so the default tier branches would show 0.0° / Rep ? noise.
    template_3d = parsed.get("template", "")
    _3D_TEMPLATES = {
        "tier_a_pose_class",
        "tier_a_motion_plane",
        "tier_a_active_side_3d",
        "tier_a_neck_rotation",
        "tier_a_neck_flexion_3d",
        "tier_a_trunk_axial_yaw_rom",
        "tier_a_trunk_rotation_direction",
        "tier_a_trunk_lean_direction",
        "tier_a_trunk_sagittal_direction",
        "tier_a_trunk_sagittal_rom",
        "tier_a_hip_flexion_3d",
        "tier_a_hip_abduction_rom_3d",
        "tier_a_hip_hinge_rom_3d",
        "tier_a_hip_extension_3d",
        "tier_a_hip_hyperextension_3d",
        "tier_a_prone_arm_lift_3d",
        "tier_a_prone_press_up_3d",
        "tier_a_shoulder_extension_3d",
        "tier_a_knee_pushup_3d",
        "tier_a_quad_stretch_3d",
        "tier_a_quad_stretch_depth_3d",
        "tier_a_trunk_side_bend_3d",
        "tier_a_sidelying_abduction_3d",
        "tier_a_standing_row_3d",
        "tier_a_limb_extension_arm_3d",
        "tier_a_limb_extension_leg_3d",
        "tier_a_knee_flexion_3d",
        "tier_a_lower_body_depth_3d",
        "tier_a_shoulder_er_3d",
        "tier_a_shoulder_elevation_rom_3d",
        "tier_a_elbow_flexion_rom_3d",
        "tier_a_wrist_rom_3d",
        "tier_a_trunk_stability_hold_3d",
        "tier_a_plank_back_bend_3d",
        "tier_b_axial_vs_lean_comparison",
        "tier_b_compensatory_3d",
        "tier_c_coordination_3d",
    }
    if template_3d in _3D_TEMPLATES:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        # Surface the DATA SOURCE so reviewers know whether the metric came
        # from 2D VitPose keypoints or SAM-3D-Body. Uses the same
        # authoritative template→source classification as the Keypoint
        # Source filter, so the box and the filter always agree.
        _src_code = _resolve_kp_source_for_sample(parsed)
        _src = {
            "sam3dbody": "SAM-3D-Body (3D keypoints / joint angles)",
            "vitpose": "2D VitPose keypoints",
            "blazepose": "BlazePose keypoints",
        }.get(_src_code, _src_code)
        lines.append(f"| Source | {_src} |")
        # Surface the rep_index for all 3D Tier A so reviewers know which
        # rep the answer describes. Tier B/C report reps in their own rows.
        ri_disp = v.get("rep_index")
        if ri_disp is not None and template_3d.startswith("tier_a_"):
            # Disambiguate: the rep_index stored in the metric is the
            # filesystem rep dir number (where `repetition_0` is the first).
            # The rest of the app shows 1-based display numbers (Rep 1 = the
            # first rep). Show BOTH so reviewers can match the metric to the
            # right rep in the dropdown. display Rep N = filesystem rep N-1.
            try:
                _fs = int(ri_disp)
                lines.append(f"| Rep | filesystem `repetition_{_fs}` (= display Rep {_fs + 1}) |")
            except (ValueError, TypeError):
                lines.append(f"| Rep | {ri_disp} |")
        if template_3d == "tier_a_pose_class":
            lines.append(f"| Pose Class | **{v.get('pose_class', '?')}** |")
            # Hide pose_class_confidence: it's a heuristic margin score
            # (0.5–0.95) from _pose_class geometric thresholds, not a learned
            # probability, so the number is more confusing than useful.
            lines.append(f"| Quality Verdict | {v.get('quality_verdict', '?')} |")
        elif template_3d == "tier_a_motion_plane":
            lines.append(f"| Motion Plane | **{v.get('motion_plane', '?')}** |")
            # Hide motion_plane_confidence (dominant-axis variance) — also a
            # heuristic score rather than a probability.
        elif template_3d == "tier_a_active_side_3d":
            lines.append(f"| Active Side | **{v.get('active_side', '?')}** |")
            rom = v.get("rom_by_joint") or {}
            if rom:
                lines.append("")
                lines.append("**ROM per joint (3D):**\n")
                lines.append("| Joint | ROM (°) |")
                lines.append("|-------|---------|")
                for jk, jv in rom.items():
                    if jv is None:
                        continue
                    lines.append(f"| {jk.replace('_', ' ')} | {float(jv):.1f}° |")
        elif template_3d == "tier_a_neck_rotation":
            rng = v.get("neck_rotation_range_deg")
            lines.append(f"| Neck Rotation Range | {rng:.1f}° |" if rng is not None else "| Neck Rotation Range | ? |")
            lines.append(f"| Direction | **{v.get('neck_rotation_direction', '?')}** |")
            pf = v.get("neck_rotation_peak_frame")
            if pf is not None:
                lines.append(f"| Peak Frame | {pf} |")
        elif template_3d == "tier_a_neck_flexion_3d":
            mo = v.get("neck_motion")
            if mo:
                lines.append(f"| Motion | **{mo}** |")
            rng = v.get("neck_range_deg")
            if rng is not None:
                lines.append(f"| Neck Range | **{float(rng):.1f}°** |")
            lines.append(f"| Direction | **{v.get('neck_direction', '?')}** |")
            bn = v.get("correct_bin")
            if bn:
                lines.append(f"| ROM Bin | {bn} |")
            fr = v.get("flex_range_deg")
            sr = v.get("sidebend_range_deg")
            if fr is not None:
                lines.append(f"| Flex/ext range | {float(fr):.1f}° |")
            if sr is not None:
                lines.append(f"| Side-bend range | {float(sr):.1f}° |")
        elif template_3d in ("tier_a_trunk_axial_yaw_rom", "tier_a_trunk_rotation_direction"):
            rng = v.get("trunk_axial_yaw_range_deg")
            lines.append(f"| Axial Yaw Range | {rng:.1f}° |" if rng is not None else "| Axial Yaw Range | ? |")
            lines.append(f"| Direction | **{v.get('trunk_axial_yaw_direction', '?')}** |")
            pf = v.get("trunk_axial_yaw_peak_frame")
            if pf is not None:
                lines.append(f"| Peak Frame | {pf} |")
            method = v.get("computation_method")
            if method:
                lines.append(f"| Method | `{method}` |")
        elif template_3d == "tier_a_trunk_lean_direction":
            lines.append(f"| Lateral Direction | **{v.get('lateral_direction', '?')}** |")
            lmx = v.get("lateral_max_deg")
            lmn = v.get("lateral_mean_deg")
            if lmx is not None:
                lines.append(f"| Lateral Max | {lmx:.1f}° |")
            if lmn is not None:
                lines.append(f"| Lateral Mean | {lmn:.1f}° |")
        elif template_3d == "tier_a_trunk_sagittal_direction":
            lines.append(f"| Directionality | **{v.get('sagittal_directionality', '?')}** |")
            peak = v.get("sagittal_peak_sign")
            if peak:
                lines.append(f"| Peak-Frame Sign | {peak} |")
            smx = v.get("sagittal_max_deg")
            smn = v.get("sagittal_mean_deg")
            if smx is not None:
                lines.append(f"| Sagittal Max | {float(smx):.1f}° |")
            if smn is not None:
                lines.append(f"| Sagittal Mean | {float(smn):.1f}° |")
        elif template_3d == "tier_a_trunk_sagittal_rom":
            smx = v.get("sagittal_max_deg")
            if smx is not None:
                lines.append(f"| Sagittal Max | **{float(smx):.1f}°** |")
            smn = v.get("sagittal_mean_deg")
            if smn is not None:
                lines.append(f"| Sagittal Mean | {float(smn):.1f}° |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| ROM Bin | {bin_label} |")
        elif template_3d == "tier_b_axial_vs_lean_comparison":
            ria = v.get("rep_a", "?")
            rib = v.get("rep_b", "?")
            ya = v.get("rep_a_yaw_range_deg")
            yb = v.get("rep_b_yaw_range_deg")
            diff = v.get("yaw_diff_deg")
            lines.append(f"| Reps Compared | rep {ria} vs rep {rib} |")
            if ya is not None:
                lines.append(f"| Rep {ria} Axial Yaw Range | {float(ya):.1f}° |")
            if yb is not None:
                lines.append(f"| Rep {rib} Axial Yaw Range | {float(yb):.1f}° |")
            if diff is not None:
                direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                lines.append(f"| Difference (A − B) | {direction} {abs(float(diff)):.1f}° |")
        elif template_3d == "tier_b_compensatory_3d":
            ria = v.get("rep_a", "?")
            rib = v.get("rep_b", "?")
            la = v.get("rep_a_lean_from_upright_deg")
            lb = v.get("rep_b_lean_from_upright_deg")
            sa = v.get("rep_a_sign")
            sb = v.get("rep_b_sign")
            diff = v.get("lean_diff_deg")
            lines.append(f"| Reps Compared | rep {ria} vs rep {rib} |")
            if la is not None:
                sa_str = f" ({sa})" if sa else ""
                lines.append(f"| Rep {ria} Lean from Upright | {float(la):.1f}°{sa_str} |")
            if lb is not None:
                sb_str = f" ({sb})" if sb else ""
                lines.append(f"| Rep {rib} Lean from Upright | {float(lb):.1f}°{sb_str} |")
            if diff is not None:
                direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                lines.append(f"| Difference (A − B) | {direction} {abs(float(diff)):.1f}° |")
        elif template_3d == "tier_a_lower_body_depth_3d":
            krom = v.get("knee_rom_deg")
            if krom is not None:
                lines.append(f"| Knee ROM | **{float(krom):.1f}°** |")
            side = v.get("side_used")
            if side:
                lines.append(f"| Side Used | **{side}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Depth Bin | {bin_label} |")
            for s, key in (("left", "raw_left_knee_rom"), ("right", "raw_right_knee_rom")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| Raw {s} knee ROM | {float(val):.1f}° |")
        elif template_3d == "tier_a_shoulder_er_3d":
            mrom = v.get("max_rom_deg")
            if mrom is not None:
                lines.append(f"| Shoulder ER ROM | **{float(mrom):.1f}°** |")
            side = v.get("side_used")
            if side:
                lines.append(f"| Side Used | **{side}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| ROM Bin | {bin_label} |")
            for s, key in (("left", "raw_left_rom_deg"), ("right", "raw_right_rom_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| Raw {s} ROM | {float(val):.1f}° |")
            named = v.get("side_named_in_question")
            if named is not None:
                lines.append(f"| Side named in Q | {'yes' if named else 'no'} |")
        elif template_3d == "tier_a_shoulder_elevation_rom_3d":
            elev = v.get("shoulder_elevation_peak")
            if elev is not None:
                lines.append(f"| Elevation peak (×trunk) | **{float(elev):.2f}** |")
            side = v.get("side_used")
            if side:
                lines.append(f"| Side Used | **{side}** |")
            raw3d = v.get("side_used_raw_3d")
            if raw3d and raw3d != side:
                lines.append(f"| Raw 3D side (pre-swap) | {raw3d} |")
            if v.get("vitpose_swap_applied"):
                lines.append(f"| VitPose swap applied | yes |")
            sres = v.get("side_resolution")
            if sres:
                lines.append(f"| Side resolution | {sres} |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Elevation Bin | {bin_label} |")
            for s, key in (("left", "raw_left_peak"), ("right", "raw_right_peak")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| Raw {s} peak | {float(val):.2f} |")
            named = v.get("side_named_in_question")
            if named is not None:
                lines.append(f"| Side named in Q | {'yes' if named else 'no'} |")
        elif template_3d == "tier_a_elbow_flexion_rom_3d":
            erom = v.get("elbow_peak_flexion_deg")
            if erom is None:
                erom = v.get("elbow_rom_deg")  # legacy key
            if erom is not None:
                lines.append(f"| Elbow Peak Flexion | **{float(erom):.1f}°** |")
            side = v.get("side_used")
            if side:
                lines.append(f"| Side Used | **{side}** |")
            raw3d = v.get("side_used_raw_3d")
            if raw3d and raw3d != side:
                lines.append(f"| Raw 3D side (pre-swap) | {raw3d} |")
            if v.get("vitpose_swap_applied"):
                lines.append(f"| VitPose swap applied | yes |")
            sres = v.get("side_resolution")
            if sres:
                lines.append(f"| Side resolution | {sres} |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Flexion Bin | {bin_label} |")
            for s in ("left", "right"):
                pf = v.get(f"raw_{s}_peak_flexion_deg")
                if pf is not None:
                    lines.append(f"| Raw {s} peak flexion | {float(pf):.1f}° |")
            named = v.get("side_named_in_question")
            if named is not None:
                lines.append(f"| Side named in Q | {'yes' if named else 'no'} |")
        elif template_3d == "tier_a_hip_abduction_rom_3d":
            arom = v.get("hip_abduction_rom_deg")
            if arom is not None:
                lines.append(f"| Hip Abduction ROM | **{float(arom):.1f}°** |")
            side = v.get("side_used")
            if side:
                lines.append(f"| Side Used | **{side}** |")
            raw3d = v.get("side_used_raw_3d")
            if raw3d and raw3d != side:
                lines.append(f"| Raw 3D side (pre-swap) | {raw3d} |")
            if v.get("vitpose_swap_applied"):
                lines.append(f"| VitPose swap applied | yes |")
            sres = v.get("side_resolution")
            if sres:
                lines.append(f"| Side resolution | {sres} |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| ROM Bin | {bin_label} |")
            for s, key in (("left", "raw_left_rom_deg"), ("right", "raw_right_rom_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| Raw {s} ROM | {float(val):.1f}° |")
            named = v.get("side_named_in_question")
            if named is not None:
                lines.append(f"| Side named in Q | {'yes' if named else 'no'} |")
        elif template_3d == "tier_a_hip_hinge_rom_3d":
            hp = v.get("hip_hinge_peak_deg")
            if hp is not None:
                lines.append(f"| Hip Hinge Depth | **{float(hp):.1f}°** |")
            ss = v.get("stance_side")
            if ss:
                lines.append(f"| Stance Side | **{ss}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Hinge Bin | {bin_label} |")
            for s, key in (("left", "left_peak_deg"), ("right", "right_peak_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| Raw {s} hinge | {float(val):.1f}° |")
        elif template_3d == "tier_a_hip_extension_3d":
            hp = v.get("hip_extension_peak_deg")
            if hp is not None:
                lines.append(f"| Hip Extension Peak (5° grid) | **{float(hp):.0f}°** |")
            raw = v.get("hip_extension_peak_raw_deg")
            if raw is not None:
                lines.append(f"| Raw peak (pre-rounding) | {float(raw):.1f}° |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Extension Bin | {bin_label} |")
            for s, key in (("left", "left_peak_deg"), ("right", "right_peak_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| Raw {s} hip angle | {float(val):.1f}° |")
        elif template_3d == "tier_a_hip_hyperextension_3d":
            rom = v.get("hip_hyperext_rom_deg")
            if rom is not None:
                lines.append(f"| Hip-Extension ROM | **{float(rom):.1f}°** |")
            ms = v.get("moving_side")
            if ms:
                lines.append(f"| Moving Side | **{ms}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| ROM Bin | {bin_label} |")
            for s, key in (("left", "left_rom_deg"), ("right", "right_rom_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| Raw {s} hip-ext ROM | {float(val):.1f}° |")
        elif template_3d == "tier_a_prone_arm_lift_3d":
            rom = v.get("prone_arm_lift_rom")
            if rom is not None:
                lines.append(f"| Arm-Lift ROM (frac. arm length) | **{float(rom):.3f}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Lift Bin | {bin_label} |")
            for lbl, key in (("rest", "prone_arm_lift_rest"), ("peak", "prone_arm_lift_peak")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| {lbl} wrist height | {float(val):.3f} |")
        elif template_3d == "tier_a_prone_press_up_3d":
            rom = v.get("prone_press_up_rom_deg")
            if rom is not None:
                lines.append(f"| Press-Up Elbow ROM | **{float(rom):.1f}°** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Press-Up Bin | {bin_label} |")
            for lbl, key in (("rest", "prone_press_up_rest_deg"), ("peak", "prone_press_up_peak_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| {lbl} elbow angle | {float(val):.1f}° |")
        elif template_3d == "tier_a_shoulder_extension_3d":
            rom = v.get("shoulder_extension_rom_deg")
            if rom is not None:
                lines.append(f"| Shoulder Extension ROM (5° grid) | **{float(rom):.0f}°** |")
            raw = v.get("shoulder_extension_rom_raw_deg")
            if raw is not None:
                lines.append(f"| Raw ROM (pre-rounding) | {float(raw):.1f}° |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Extension Bin | {bin_label} |")
            for lbl, key in (("rest", "shoulder_extension_rest_deg"), ("peak", "shoulder_extension_peak_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| {lbl} arm angle | {float(val):.1f}° |")
        elif template_3d == "tier_a_knee_pushup_3d":
            # Reframed to peak elbow flexion at the bottom. Falls back to the
            # legacy ROM fields for older samples (backwards compatible).
            pf = v.get("knee_pushup_peak_flexion_deg")
            if pf is not None:
                lines.append(f"| Push-Up Peak Elbow Flexion (5° grid) | **{float(pf):.0f}°** |")
                raw = v.get("knee_pushup_peak_flexion_raw_deg")
                if raw is not None:
                    lines.append(f"| Raw peak flexion (pre-rounding) | {float(raw):.1f}° |")
                bot = v.get("knee_pushup_bottom_angle_deg")
                if bot is not None:
                    lines.append(f"| Bottom elbow angle (most bent) | {float(bot):.1f}° |")
            else:
                rom = v.get("knee_pushup_rom_deg")
                if rom is not None:
                    lines.append(f"| Push-Up Elbow ROM | **{float(rom):.1f}°** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Push-Up Bin | {bin_label} |")
        elif template_3d == "tier_a_quad_stretch_3d":
            rom = v.get("quad_stretch_rom_deg")
            if rom is not None:
                lines.append(f"| Quad-Stretch Knee ROM (5° grid) | **{float(rom):.0f}°** |")
            raw = v.get("quad_stretch_rom_raw_deg")
            if raw is not None:
                lines.append(f"| Raw ROM (pre-rounding) | {float(raw):.1f}° |")
            ms = v.get("moving_side")
            if ms:
                lines.append(f"| Moving (stretched) leg | **{ms}** |")
            mn = v.get("quad_stretch_min_deg")
            if mn is not None:
                lines.append(f"| Most-flexed knee angle | {float(mn):.1f}° |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Stretch Bin | {bin_label} |")
            for s in ("left", "right"):
                val = v.get(f"{s}_rom_deg")
                if val is not None:
                    lines.append(f"| Raw {s} knee ROM | {float(val):.1f}° |")
        elif template_3d == "tier_a_quad_stretch_depth_3d":
            pf = v.get("quad_stretch_peak_flexion_deg")
            if pf is not None:
                lines.append(f"| Quad-Stretch Peak Knee Flexion (5° grid) | **{float(pf):.0f}°** |")
            raw = v.get("quad_stretch_peak_flexion_raw_deg")
            if raw is not None:
                lines.append(f"| Raw peak flexion (pre-rounding) | {float(raw):.1f}° |")
            mn = v.get("quad_stretch_min_angle_deg")
            if mn is not None:
                lines.append(f"| Most-flexed knee angle | {float(mn):.1f}° |")
            ms = v.get("moving_side")
            if ms:
                lines.append(f"| Stretched leg (named side) | **{ms}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Depth Bin | {bin_label} |")
        elif template_3d == "tier_a_standing_row_3d":
            # Reframed to peak elbow flexion at the pulled-in position.
            # Falls back to legacy ROM fields for older samples.
            pf = v.get("standing_row_peak_flexion_deg")
            if pf is not None:
                lines.append(f"| Row Peak Elbow Flexion (5° grid) | **{float(pf):.0f}°** |")
                raw = v.get("standing_row_peak_flexion_raw_deg")
                if raw is not None:
                    lines.append(f"| Raw peak flexion (pre-rounding) | {float(raw):.1f}° |")
                pin = v.get("standing_row_pulled_in_angle_deg")
                if pin is not None:
                    lines.append(f"| Pulled-in elbow angle (most bent) | {float(pin):.1f}° |")
            else:
                rom = v.get("standing_row_rom_deg")
                if rom is not None:
                    lines.append(f"| Row Elbow ROM | **{float(rom):.1f}°** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Row Bin | {bin_label} |")
        elif template_3d == "tier_a_trunk_side_bend_3d":
            rom = v.get("trunk_side_bend_rom_deg")
            if rom is not None:
                lines.append(f"| Trunk Side-Bend ROM (5° grid) | **{float(rom):.0f}°** |")
            raw = v.get("trunk_side_bend_rom_raw_deg")
            if raw is not None:
                lines.append(f"| Raw ROM (pre-rounding) | {float(raw):.1f}° |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Side-Bend Bin | {bin_label} |")
            for lbl, key in (("rest", "trunk_side_bend_rest_deg"), ("peak", "trunk_side_bend_peak_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| {lbl} trunk lean | {float(val):.1f}° |")
        elif template_3d == "tier_a_sidelying_abduction_3d":
            rom = v.get("sidelying_abduction_rom_deg")
            if rom is not None:
                lines.append(f"| Hip Abduction ROM (5° grid) | **{float(rom):.0f}°** |")
            raw = v.get("sidelying_abduction_rom_raw_deg")
            if raw is not None:
                lines.append(f"| Raw ROM (pre-rounding) | {float(raw):.1f}° |")
            ms = v.get("moving_side")
            if ms:
                lines.append(f"| Moving (lifted) leg | **{ms}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Lift Bin | {bin_label} |")
            for s in ("left", "right"):
                val = v.get(f"{s}_rom_deg")
                if val is not None:
                    lines.append(f"| Raw {s} hip ROM | {float(val):.1f}° |")
        elif template_3d in ("tier_a_limb_extension_arm_3d",
                             "tier_a_limb_extension_leg_3d"):
            mode = v.get("limb_extension_mode", "peak")
            measured = v.get("limb_extension_measured_deg")
            if measured is None:  # pre-mode samples — fall back to peak
                measured = v.get("limb_extension_peak_deg")
            row_label = "Limb Extension ROM" if mode == "rom" else "Limb Extension Peak"
            if measured is not None:
                lines.append(f"| {row_label} | **{float(measured):.1f}°** |")
            lr = v.get("limb_reported")
            if lr:
                lines.append(f"| Limb Reported | **{lr}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| {'ROM Bin' if mode == 'rom' else 'Extension Bin'} | {bin_label} |")
            # Show both peak and ROM for both limbs as context.
            ap = v.get("arm_peak_deg")
            if ap is not None:
                ar = v.get("arm_rom_deg")
                extra = f", ROM {float(ar):.1f}°" if ar is not None else ""
                lines.append(f"| Arm (either side, {v.get('arm_peak_side','?')}) | peak {float(ap):.1f}°{extra} |")
            lpk = v.get("leg_peak_deg")
            if lpk is not None:
                lr2 = v.get("leg_rom_deg")
                extra = f", ROM {float(lr2):.1f}°" if lr2 is not None else ""
                lines.append(f"| Leg (either side, {v.get('leg_peak_side','?')}) | peak {float(lpk):.1f}°{extra} |")
        elif template_3d == "tier_a_wrist_rom_3d":
            adir = v.get("active_direction")
            arom = v.get("active_rom_deg")
            if adir is not None and arom is not None:
                lines.append(f"| Active direction | **{adir}** ({float(arom):.1f}°) |")
            flex = v.get("flexion_rom_deg")
            if flex is not None:
                lines.append(f"| Flexion ROM | {float(flex):.1f}° |")
            ext = v.get("extension_rom_deg")
            if ext is not None:
                lines.append(f"| Extension ROM | {float(ext):.1f}° |")
            total = v.get("total_span_deg")
            if total is not None:
                lines.append(f"| Total span | {float(total):.1f}° |")
            side = v.get("side_used")
            if side:
                lines.append(f"| Side Used | **{side}** |")
            sig = v.get("flexion_sign")
            if sig:
                lines.append(f"| Flexion direction sign | {sig} (camera-relative) |")
            src = v.get("signal_source")
            if src:
                lines.append(f"| Signal source | `{src}` |")
            for s, key in (("left", "raw_left_total_deg"), ("right", "raw_right_total_deg")):
                val = v.get(key)
                if val is not None:
                    lines.append(f"| Raw {s} total | {float(val):.1f}° |")
            named = v.get("side_named_in_question")
            if named is not None:
                lines.append(f"| Side named in Q | {'yes' if named else 'no'} |")
        elif template_3d in ("tier_a_hip_flexion_3d", "tier_a_knee_flexion_3d"):
            joint_word = "Hip" if template_3d == "tier_a_hip_flexion_3d" else "Knee"
            peak_key = "peak_hip_flexion_deg" if template_3d == "tier_a_hip_flexion_3d" else "peak_knee_flexion_deg"
            pf = v.get(peak_key)
            if pf is not None:
                lines.append(f"| Peak {joint_word} Flexion | **{float(pf):.1f}°** |")
            side = v.get("side_used")
            if side:
                lines.append(f"| Active Side | **{side}** |")
            bin_label = v.get("correct_bin")
            if bin_label:
                lines.append(f"| Bin | {bin_label} |")
            for s in ("left", "right"):
                raw = v.get(f"raw_{s}")
                if raw is None:
                    continue
                pm = raw.get("peak_max_deg")
                rom = raw.get("rom_deg")
                ma = raw.get("min_angle_deg")
                lines.append(
                    f"| Raw {s} | max={float(pm):.1f}° ROM={float(rom):.1f}° min={float(ma):.1f}° |"
                )
        elif template_3d == "tier_a_trunk_stability_hold_3d":
            std = v.get("stability_std_drift_deg")
            if std is not None:
                lines.append(f"| Std drift | **{float(std):.2f}°** |")
            mx = v.get("stability_max_drift_deg")
            if mx is not None:
                lines.append(f"| Max drift | {float(mx):.2f}° |")
            p90 = v.get("stability_p90_drift_deg")
            if p90 is not None:
                lines.append(f"| P90 drift | {float(p90):.2f}° |")
            nf = v.get("stability_n_frames")
            if nf is not None:
                lines.append(f"| Frames used | {nf} |")
            bin_label = v.get("stability_bin")
            if bin_label:
                lines.append(f"| Bin | `{bin_label}` |")
        elif template_3d == "tier_a_plank_back_bend_3d":
            mn = v.get("back_bend_mean_deg")
            if mn is not None:
                lines.append(f"| Mean bend | **{float(mn):.1f}°** |")
            sd = v.get("back_bend_std_deg")
            if sd is not None:
                lines.append(f"| Std bend | {float(sd):.2f}° |")
            p90 = v.get("back_bend_p90_deg")
            if p90 is not None:
                lines.append(f"| P90 bend | {float(p90):.1f}° |")
            mx = v.get("back_bend_max_deg")
            if mx is not None:
                lines.append(f"| Max bend | {float(mx):.1f}° |")
            nf = v.get("back_bend_n_frames")
            if nf is not None:
                lines.append(f"| Frames used | {nf} |")
            bin_label = v.get("back_bend_bin")
            if bin_label:
                lines.append(f"| Bin | `{bin_label}` |")
        elif template_3d == "tier_c_coordination_3d":
            lines.append(f"| Joint Part | **{v.get('joint_part', '?')}** |")
            ma = v.get("mean_asymmetry_deg")
            mx = v.get("max_asymmetry_deg")
            if ma is not None:
                lines.append(f"| Mean Asymmetry | {ma:.2f}° |")
            if mx is not None:
                lines.append(f"| Max Asymmetry | {mx:.2f}° |")
            asym = v.get("asym_by_rep") or {}
            if asym:
                lines.append("")
                lines.append("**Asymmetry per rep:**\n")
                lines.append("| Rep | Asymmetry (°) |")
                lines.append("|-----|---------------|")
                for ri, av in sorted(asym.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0):
                    lines.append(f"| {ri} | {float(av):.2f}° |")
            lines.append(f"| Reps Analysed | {v.get('n_reps_analyzed', '?')} |")
        # Human-readable definition of how the metric was computed, surfaced
        # below the verification table for any 3D template that supplies it
        # in metadata.computation_explanation. Kept out of the question text
        # to avoid biasing the model with the geometric recipe.
        ce = (parsed.get("metadata") or {}).get("computation_explanation")
        if not ce:
            ce = parsed.get("computation_explanation")
        if ce:
            lines.append("")
            lines.append("**How this is computed:**")
            lines.append(f"> {ce}")
        return "\n".join(lines)

    if "TIER_A" in tier:
        method = v.get("computation_method", "")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Rep | {v.get('rep_index', '?')} |")
        if method == "hold_duration":
            lines.append(f"| Hold Frames | {v.get('hold_frames', '?')} |")
            lines.append(f"| Total Frames | {v.get('num_frames', '?')} |")
            lines.append(f"| FPS | {v.get('fps', 0):.2f} |")
            lines.append(f"| Hold Time | {v.get('hold_time_seconds', 0):.2f} s |")
        elif method in ("temporal_grounding", "phase_identification"):
            nf = v.get("num_frames") or 0
            pf = v.get("peak_frame") or v.get("movement_frame") or 0
            pct = int(pf / max(nf, 1) * 100)
            lines.append(f"| Peak Frame | {pf} / {nf} (~{pct}%) |")
            if method == "phase_identification":
                lines.append(f"| Peak Ratio | {v.get('peak_ratio') or 0:.3f} |")
        else:
            lines.append(f"| ROM | {v.get('rom_degrees') or 0:.1f}° |")
            lines.append(f"| Peak Angle | {v.get('peak_angle_degrees') or 0:.1f}° |")
        lines.append(f"| Reps Analysed | 1 |")
        # Show contralateral joint metrics when D3 used the opposite side
        _contra = v.get("d3_contralateral_joint")
        if _contra:
            _cj = _contra.get("joint", "?").replace("_", " ")
            lines.append(f"\n**D3 contralateral joint (actual values)**: {_cj}\n")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Rep | {_contra.get('rep_index', '?')} |")
            if method in ("temporal_grounding", "phase_identification"):
                _cnf = _contra.get("num_frames") or 0
                _cpf = _contra.get("peak_frame") or _contra.get("movement_frame") or 0
                _cpct = int(_cpf / max(_cnf, 1) * 100)
                lines.append(f"| Peak Frame | {_cpf} / {_cnf} (~{_cpct}%) |")
            else:
                if _contra.get("rom_degrees") is not None:
                    lines.append(f"| ROM | {_contra['rom_degrees']:.1f}° |")
                if _contra.get("peak_angle_degrees") is not None:
                    lines.append(f"| Peak Angle | {_contra['peak_angle_degrees']:.1f}° |")

    elif "TIER_B" in tier:
        method = v.get("computation_method", "")
        if method == "correctness_evaluation":
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Met | {'Yes' if v.get('criterion_met') else 'No'} |")
            lines.append(f"| Evidence | {_md_cell(v.get('evidence') or '?')} |")
            lines.append(f"| Check Type | {_md_cell(v.get('check_type') or '?')} |")
            lines.append(f"| Reps Analysed | {v.get('window_size') or len(v.get('window_reps') or []) or '?'} |")
        elif method == "compensatory_analysis":
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            comp_joint = v.get("compensatory_joint")
            if comp_joint:
                lines.append(f"| Comp. Joint | {comp_joint.replace('_', ' ')} |")
            primary = v.get("primary_joints")
            if primary:
                pj = ", ".join(j.replace("_", " ") for j in primary) if isinstance(primary, list) else str(primary)
                lines.append(f"| Primary Joints | {pj} |")
            if v.get("mean_rom_deg") is not None:
                lines.append(f"| Mean ROM | {v['mean_rom_deg']:.1f}° |")
            if v.get("cov_pct") is not None:
                lines.append(f"| CoV | {v['cov_pct']:.1f}% |")
            lines.append(f"| Reps Analysed | {v.get('window_size') or len(v.get('window_reps') or []) or '?'} |")
            # Show contralateral joint metrics when D1 used the opposite side
            _contra = v.get("d1_contralateral_joint")
            if _contra:
                _cj = _contra.get("joint", "?").replace("_", " ")
                lines.append(f"\n**D1 contralateral joint (actual values)**: {_cj}\n")
                lines.append(f"| Metric | Value |")
                lines.append(f"|--------|-------|")
                if _contra.get("mean_rom_deg") is not None:
                    lines.append(f"| Mean ROM | {_contra['mean_rom_deg']:.1f}° |")
                if _contra.get("cov_pct") is not None:
                    lines.append(f"| CoV | {_contra['cov_pct']:.1f}% |")
        elif method == "error_detection":
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Error | {_md_cell(v.get('error_category') or '?')} |")
            lines.append(f"| Detected | {'Yes' if v.get('error_detected') else 'No'} |")
            lines.append(f"| Evidence | {_md_cell(v.get('evidence') or '?')} |")
            if v.get("left_rom_deg") is not None:
                lines.append(f"| L ROM | {v['left_rom_deg']:.1f}° |")
                lines.append(f"| R ROM | {v.get('right_rom_deg') or 0:.1f}° |")
                lines.append(f"| L/R Diff | {v.get('lr_diff_pct') or 0:.1f}% |")
            lines.append(f"| Reps Analysed | {v.get('window_size') or len(v.get('window_reps') or []) or '?'} |")
        else:
            rc = parsed.get("rep_comparison", {})
            template_b = parsed.get("template", "")
            lines.append(f"| Metric | Rep {rc.get('rep_a', '?')} | Rep {rc.get('rep_b', '?')} |")
            lines.append(f"|--------|---------|---------|")
            if template_b == "tier_b_peak_comparison":
                a_val = v.get("rep_a_peak_deg") or 0
                b_val = v.get("rep_b_peak_deg") or 0
                lines.append(f"| Peak Angle | {a_val:.1f}° | {b_val:.1f}° |")
                diff = v.get("peak_diff_deg") or 0
                pct = abs(diff / a_val * 100) if abs(a_val) > 0.1 else 0
                direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                lines.append(f"| Change | {direction} {abs(diff):.1f}° ({pct:.0f}%) ||")
            else:
                a_val = v.get("rep_a_rom_deg") or 0
                b_val = v.get("rep_b_rom_deg") or 0
                lines.append(f"| ROM | {a_val:.1f}° | {b_val:.1f}° |")
                diff = v.get("rom_diff_deg") or 0
                pct = abs(diff / a_val * 100) if abs(a_val) > 0.1 else 0
                direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
                lines.append(f"| Change | {direction} {abs(diff):.1f}° ({pct:.0f}%) ||")
            lines.append(f"| Reps Analysed | 2 ||")

    elif "TIER_C" in tier:
        template = parsed.get("template", "")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        if template in ("tier_c_kinematic_trend", "tier_c_trend_analysis"):
            lines.append(f"| Trend | {v.get('trend') or '?'} |")
            lines.append(f"| First ROM | {v.get('first_rom_deg') or 0:.1f}° |")
            lines.append(f"| Last ROM | {v.get('last_rom_deg') or 0:.1f}° |")
            lines.append(f"| Mean ROM | {v.get('mean_rom_deg') or 0:.1f}° |")
        else:  # tier_c_variability_analysis / tier_c_error_trend
            lines.append(f"| CoV | {v.get('cov_pct') or 0:.1f}% |")
            lines.append(f"| Mean ROM | {v.get('mean_rom_deg') or 0:.1f}° |")
        _n_analyzed = v.get("num_reps_analyzed") or len(v.get("rep_indices") or []) or "?"
        lines.append(f"| Reps Analysed | {_n_analyzed} |")
        # Show contralateral joint metrics when D3 used the opposite side
        _contra = v.get("d3_contralateral_joint")
        if _contra:
            _cj = _contra.get("joint", "?").replace("_", " ")
            lines.append(f"\n**D3 contralateral joint**: {_cj}\n")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            if template in ("tier_c_kinematic_trend", "tier_c_trend_analysis"):
                lines.append(f"| Trend | {_contra.get('trend') or '?'} |")
                lines.append(f"| First ROM | {_contra.get('first_rom_deg') or 0:.1f}° |")
                lines.append(f"| Last ROM | {_contra.get('last_rom_deg') or 0:.1f}° |")
                lines.append(f"| Mean ROM | {_contra.get('mean_rom_deg') or 0:.1f}° |")
            else:
                lines.append(f"| CoV | {_contra.get('cov_pct') or 0:.1f}% |")
                lines.append(f"| Mean ROM | {_contra.get('mean_rom_deg') or 0:.1f}° |")
            _cn = _contra.get("num_reps_analyzed") or len(_contra.get("rep_indices") or []) or "?"
            lines.append(f"| Reps Analysed | {_cn} |")

    elif "TIER_D" in tier:
        template = parsed.get("template", "")
        lines.append(f"| Metric | Left | Right |")
        lines.append(f"|--------|------|-------|")
        if template == "tier_d_side_comparison":
            lines.append(f"| Mean ROM | {v.get('left_mean_rom_deg') or 0:.1f}° | {v.get('right_mean_rom_deg') or 0:.1f}° |")
            l_count = v.get('left_rep_count') or 0
            r_count = v.get('right_rep_count') or 0
            lines.append(f"| Reps Analysed | {l_count} | {r_count} |")
        elif template == "tier_d_asymmetry_detection":
            lines.append(f"| Mean ROM | {v.get('left_mean_rom_deg') or 0:.1f}° | {v.get('right_mean_rom_deg') or 0:.1f}° |")
            asym = v.get('asymmetry_pct') or 0
            sym = v.get('is_symmetric')
            lines.append(f"| Asymmetry | {asym:.1f}% ({'symmetric' if sym else 'asymmetric'}) ||")
            lines.append(f"| Higher Side | {v.get('higher_side') or '?'} ||")
            _n = v.get('num_reps_analyzed') or len(v.get('rep_indices') or []) or '?'
            lines.append(f"| Reps Analysed | {_n} ||")
        elif template == "tier_d_side_consistency":
            lines.append(f"| CoV | {v.get('left_cov_pct') or 0:.1f}% | {v.get('right_cov_pct') or 0:.1f}% |")
            lines.append(f"| More Consistent | {v.get('more_consistent_side') or '?'} ||")
            _n = v.get('num_reps_analyzed') or len(v.get('rep_indices') or []) or '?'
            lines.append(f"| Reps Analysed | {_n} ||")
        elif template == "tier_d_peak_comparison":
            lines.append(f"| Mean Peak | {v.get('left_mean_peak_deg') or 0:.1f}° | {v.get('right_mean_peak_deg') or 0:.1f}° |")
            lines.append(f"| Higher Side | {v.get('higher_side') or '?'} ||")
            _n = v.get('num_reps_analyzed') or len(v.get('rep_indices') or []) or '?'
            lines.append(f"| Reps Analysed | {_n} ||")

    return "\n".join(lines)


def render_distractors(parsed: Dict) -> str:
    """Render distractor analysis."""
    # vo3d Visual-Observations records have no MCQA distractors (full block, not
    # a single multiple-choice question). The answers[] audit lives in the
    # question panel (render_vo_block); nothing to show here.
    if parsed.get("vo_answers") is not None:
        return ""
    lines = ["### Distractor Analysis\n"]
    sources = parsed["distractor_sources"]
    choices = parsed["choices"]
    correct_idx = parsed["correct_index"]
    labels = string.ascii_uppercase

    source_idx = 0
    for i, choice in enumerate(choices):
        if i == correct_idx:
            lines.append(f"**{labels[i]})** ✅ Correct answer\n")
        else:
            src = sources[source_idx] if source_idx < len(sources) else "unknown"
            lines.append(f"**{labels[i]})** ⚠️ `{src}`\n")
            source_idx += 1

    return "\n".join(lines)


def _inline_think(sample: Optional[Dict]) -> str:
    """Extract a <think>...</think> trace from a sample's own assistant turn.

    Newer reasoning datasets (e.g. mcqa_video_1405_reasoning) carry the trace
    inline in messages[-1] instead of a separate reas2 reasoning_index sidecar.
    Returns the think-block body, or '' if none."""
    if not sample:
        return ""
    for msg in reversed(sample.get("messages") or []):
        if msg.get("role") != "assistant":
            continue
        c = msg.get("content")
        if isinstance(c, list):
            c = "".join(str(it.get("text") or "") for it in c
                        if isinstance(it, dict))
        if not isinstance(c, str):
            return ""
        m = re.search(r"<think>(.*?)</think>", c, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
    return ""


def _old_reas_trace_block(sample: Optional[Dict]) -> str:
    """Render the original/source reasoning trace stashed in metadata.old_reas_trace.

    For the 1805_binary base, the source `reasoning_trace` was DROPPED from the
    training target (a misaligned different-pipeline artifact) but preserved in the
    browse JSONL under metadata.old_reas_trace so it can be eyeballed here. Returns
    '' when absent."""
    if not sample:
        return ""
    md = sample.get("metadata") or {}
    old = md.get("old_reas_trace")
    if not isinstance(old, str) or not old.strip():
        return ""
    return ("#### Original reasoning trace (source `old_reas_trace`)\n"
            "*Dropped from the SFT target — shown for reference only; "
            "NOT the trained reasoning.*\n\n"
            f"{old.strip()}")


def render_reasoning(entry: Optional[Dict], sample: Optional[Dict] = None) -> str:
    """Render a reasoning trace. Backwards compatible — three sources:

      1. The sample's own <think> block in messages[-1] (newer datasets like
         mcqa_video_1405_reasoning; preferred when present).
      2. The reas2 reasoning_index sidecar entry (`entry`), for older datasets.
      3. metadata.old_reas_trace — the original/source trace kept for reference
         (e.g. the 1805_binary source trace that was dropped from training).
    """
    old_block = _old_reas_trace_block(sample)

    # 1. Inline trace on the sample itself.
    inline = _inline_think(sample)
    if inline:
        md = (sample.get("metadata") or {}) if sample else {}
        prov = []
        teacher = md.get("reasoning_teacher_model") or md.get("reasoning_model")
        if teacher:
            prov.append(f"teacher: `{teacher}`")
        jd = md.get("reasoning_judge_decision")
        if jd:
            prov.append(f"judge: **{jd}**")
        if md.get("reasoning_regenerated"):
            prov.append("⚠️ regenerated")
        fv = md.get("frames_variant")
        if fv:
            prov.append(f"frames: `{fv}`")
        out = ["### Reasoning\n"]
        if prov:
            out.append(f"*{' · '.join(prov)}*\n")
        out.append(inline)
        if old_block:
            out.append("\n---\n")
            out.append(old_block)
        return "\n".join(out)

    # No inline <think>, but we DO have the source trace — show it on its own
    # (the 1805_binary non-reasoning base has no trained trace, only old_reas_trace).
    if old_block:
        return "### Reasoning\n\n" + old_block

    # 2. Fall back to the reas2 sidecar index entry.
    if not entry:
        return "*No reasoning trace found for this sample.*"
    think = entry.get("think", "").strip()
    if not think:
        return "*Reasoning trace present but think block is empty.*"
    judge = entry.get("judge_decision", "")
    regen = entry.get("regenerated", False)
    src_split = entry.get("source_split", "")
    provenance = []
    if judge:
        provenance.append(f"judge: **{judge}**")
    if regen:
        provenance.append("⚠️ judge-regenerated")
    if src_split:
        provenance.append(f"reas2 split: `{src_split}`")
    header = " · ".join(provenance) if provenance else ""
    lines = ["### Reasoning (reas2)\n"]
    if header:
        lines.append(f"*{header}*\n")
    lines.append(think)
    return "\n".join(lines)


def get_exercise_detail_for_video(video_id: str) -> str:
    """Get exercise detail markdown from a video_id (extracts exercise code from first segment)."""
    if EXERCISE_DF.empty:
        return "*No exercise metadata loaded*"
    exercise_code = video_id.split("_")[0]
    match = EXERCISE_DF[EXERCISE_DF["exercise_code"] == exercise_code]
    if match.empty:
        return f"*No exercise found for code {exercise_code}*"
    return render_exercise_detail(match.iloc[0])


_BULLET_RE = re.compile(r"^[\u25cf\u25cb\u2022\u25aa\u25ab\u2023\u2043\u25b8\u25e6\u2219\-\*]\s*")


def render_exercise_detail(row: pd.Series) -> str:
    """Render exercise detail markdown from CSV row."""
    lines = [f"## {row.get('exercise_name', 'Unknown Exercise')}\n"]

    position = row.get("position", "")
    muscles = row.get("muscles", "")
    joints = row.get("primary_joints", "")

    if position:
        lines.append(f"**Position**: {position}\n")
    if muscles:
        lines.append(f"**Muscles**: {muscles}\n")
    if joints:
        lines.append(f"**Primary Joints**: {joints.replace(',', ', ')}\n")

    # Phases — group pipe-delimited entries: "Name: desc" becomes a header,
    # entries starting with "-" become sub-bullets under the previous header.
    phases_raw = row.get("phases", "")
    if phases_raw:
        lines.append("---\n### Exercise Phases\n")
        for part in phases_raw.split("|"):
            part = _BULLET_RE.sub("", part.strip())
            if not part:
                continue
            if part.startswith("-"):
                # Sub-item under previous phase header
                part = part.lstrip("- ").strip()
                lines.append(f"  - {part}")
            elif ":" in part:
                name, desc = part.split(":", 1)
                desc = desc.strip()
                lines.append(f"\n**{name.strip()}**\n")
                if desc:
                    lines.append(f"{desc}\n")
            else:
                lines.append(f"{part}\n")
        lines.append("")

    # Correctness criteria
    criteria = row.get("correctness_criteria", "")
    if criteria:
        lines.append("---\n### Correctness Criteria\n")
        for c in criteria.split("|"):
            c = _BULLET_RE.sub("", c.strip())
            if c:
                lines.append(f"- {c}")
        lines.append("")

    # Typical errors
    errors = row.get("typical_errors", "")
    if errors:
        lines.append("---\n### Typical Movement Errors\n")
        for e in errors.split("|"):
            e = _BULLET_RE.sub("", e.strip())
            if not e:
                continue
            if ":" in e:
                cat, desc = e.split(":", 1)
                lines.append(f"\n**{cat.strip()}**\n")
                if desc.strip():
                    lines.append(f"{desc.strip()}\n")
            else:
                lines.append(f"- {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Statistics charts
# ---------------------------------------------------------------------------

def create_tier_chart(samples: List[Dict]) -> go.Figure:
    if not samples:
        return empty_figure("No samples loaded")

    tiers = {}
    for s in samples:
        t = s.get("metadata", {}).get("difficulty_tier", "unknown")
        tiers[t] = tiers.get(t, 0) + 1

    labels = [tier_display(t) for t in tiers]
    values = list(tiers.values())

    fig = go.Figure(go.Pie(labels=labels, values=values,
                           marker_colors=AMBER_COLORS[:len(labels)],
                           textinfo="label+value"))
    fig.update_layout(title="Tier Distribution", height=300,
                      margin=dict(t=40, b=20, l=20, r=20),
                      paper_bgcolor="white")
    return fig


def create_joint_chart(samples: List[Dict]) -> go.Figure:
    if not samples:
        return empty_figure("No samples loaded")

    joints = {}
    for s in samples:
        v = s.get("metadata", {}).get("verification", {})
        j = v.get("joint") or v.get("compensatory_joint") or "unknown"
        joints[j] = joints.get(j, 0) + 1

    sorted_joints = sorted(joints.items(), key=lambda x: x[1], reverse=True)
    names = [j.replace("_", " ") for j, _ in sorted_joints]
    counts = [c for _, c in sorted_joints]

    fig = go.Figure(go.Bar(x=counts, y=names, orientation="h",
                           marker_color=AMBER_COLORS[0]))
    fig.update_layout(title="Joint Distribution", height=300,
                      margin=dict(t=40, b=20, l=100, r=20),
                      paper_bgcolor="white", yaxis=dict(autorange="reversed"))
    return fig


def create_distractor_chart(samples: List[Dict]) -> go.Figure:
    if not samples:
        return empty_figure("No samples loaded")

    sources = {}
    for s in samples:
        for src in s.get("metadata", {}).get("distractor_sources", []):
            # Strip specific error name: "error_misattribution:pelvic_drop" -> "error_misattribution"
            key = src.split(":")[0]
            sources[key] = sources.get(key, 0) + 1

    if not sources:
        return empty_figure("No distractor data")

    sorted_src = sorted(sources.items(), key=lambda x: x[1], reverse=True)
    names = [n.replace("_", " ") for n, _ in sorted_src]
    counts = [c for _, c in sorted_src]

    fig = go.Figure(go.Bar(x=names, y=counts, marker_color=AMBER_COLORS[1]))
    fig.update_layout(title="Distractor Types", height=300,
                      margin=dict(t=40, b=60, l=40, r=20),
                      paper_bgcolor="white")
    return fig


def create_exercise_type_chart(samples: List[Dict]) -> go.Figure:
    """Stacked bar: templates per exercise type."""
    if not samples:
        return empty_figure("No samples loaded")

    from collections import defaultdict
    ex_type_template: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in samples:
        meta = s.get("metadata", {})
        ex_type = meta.get("exercise_type") or "unknown"
        template = meta.get("question_template") or "unknown"
        ex_type_template[ex_type][template] += 1

    # Collect all templates in a consistent order (by total count across all types)
    template_totals: Dict[str, int] = defaultdict(int)
    for tmpl_counts in ex_type_template.values():
        for t, c in tmpl_counts.items():
            template_totals[t] += c
    all_templates = sorted(template_totals, key=lambda t: template_totals[t], reverse=True)

    # Exercise types sorted by total sample count descending
    ex_types = sorted(ex_type_template, key=lambda e: sum(ex_type_template[e].values()), reverse=True)

    # Colour palette: one colour per template
    TEMPLATE_COLORS = [
        "#D97706", "#F59E0B", "#FCD34D", "#92400E", "#B45309", "#FBBF24",
        "#6B7280", "#9CA3AF", "#374151", "#D1D5DB", "#1F2937", "#E5E7EB",
        "#065F46", "#10B981",
    ]

    traces = []
    for i, tmpl in enumerate(all_templates):
        counts = [ex_type_template[et].get(tmpl, 0) for et in ex_types]
        if sum(counts) == 0:
            continue
        display = tmpl.replace("tier_a_", "A: ").replace("tier_b_", "B: ").replace("tier_c_", "C: ").replace("tier_d_", "D: ").replace("_", " ")
        traces.append(go.Bar(
            name=display,
            x=ex_types,
            y=counts,
            marker_color=TEMPLATE_COLORS[i % len(TEMPLATE_COLORS)],
        ))

    if not traces:
        return empty_figure("No data")

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="stack",
        title="Templates per Exercise Type",
        height=350,
        margin=dict(t=40, b=60, l=40, r=20),
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        xaxis=dict(title="Exercise Type"),
        yaxis=dict(title="Samples"),
    )
    return fig


def create_tier_per_exercise_type_chart(samples: List[Dict]) -> go.Figure:
    """Stacked bar: tiers per exercise type."""
    if not samples:
        return empty_figure("No samples loaded")

    from collections import defaultdict
    ex_type_tier: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in samples:
        meta = s.get("metadata", {})
        ex_type = meta.get("exercise_type") or "unknown"
        tier = meta.get("difficulty_tier") or "unknown"
        ex_type_tier[ex_type][tier] += 1

    tier_order = [
        "TIER_A_SINGLE_REP", "TIER_B_COMPARATIVE",
        "TIER_C_LONGITUDINAL", "TIER_D_BILATERAL",
    ]
    tier_colors = {"TIER_A_SINGLE_REP": "#D97706", "TIER_B_COMPARATIVE": "#F59E0B",
                   "TIER_C_LONGITUDINAL": "#92400E", "TIER_D_BILATERAL": "#FCD34D"}

    ex_types = sorted(ex_type_tier, key=lambda e: sum(ex_type_tier[e].values()), reverse=True)

    traces = []
    for tier in tier_order:
        counts = [ex_type_tier[et].get(tier, 0) for et in ex_types]
        if sum(counts) == 0:
            continue
        traces.append(go.Bar(
            name=tier_display(tier),
            x=ex_types,
            y=counts,
            marker_color=tier_colors.get(tier, "#6B7280"),
        ))

    if not traces:
        return empty_figure("No data")

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="stack",
        title="Tiers per Exercise Type",
        height=350,
        margin=dict(t=40, b=60, l=40, r=20),
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Exercise Type"),
        yaxis=dict(title="Samples"),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Theme & CSS
# ---------------------------------------------------------------------------

def build_theme():
    return gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="stone",
        neutral_hue="stone",
    )


CUSTOM_CSS = """
.correct-choice { border-left: 4px solid #16a34a; padding-left: 8px; }
.distractor-choice { border-left: 4px solid #D97706; padding-left: 8px; }
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# 9. UI Builder
# ---------------------------------------------------------------------------

_METHODOLOGY_DOC = """
# Dataset Methodology

This dataset trains a VLM to assess physical-therapy exercise execution from
video. Each sample is a multiple-choice question grounded in measurable
kinematic evidence from the video.

---

## 1. What each sample looks like

A sample is a `(video, question, four answer choices, correct letter)` tuple.
The question asks about one concrete observable — range of motion (ROM), peak
angle, phase timing, rep-to-rep consistency, side symmetry, compensation, and
so on. Exactly one choice is correct; the three distractors are generated from
a fixed error taxonomy (see §6) so that wrong answers correspond to realistic
clinical mistakes, not random noise.

Every sample carries a `metadata.verification` block with the raw kinematic
values that were used to compute the correct answer — this is what lets us
verify the dataset at scale and filter samples whose signal is too weak.

---

## 2. Source data

- **~10 000 patient-exercise sessions** from the Thrive capture pipeline.
  Each session records one exercise performed multiple times (reps).
- Per session, we have: the raw video, per-frame 2D pose keypoints, and a
  rep-boundary annotation (`events.json`) indicating the start/end frame of
  each repetition.
- Two keypoint sources are used depending on the body region:
  - **VitPose** (17 COCO keypoints) — whole-body exercises.
  - **BlazePose** (33 keypoints with fingertips) — hand and feet exercises.
- Metadata for every exercise (prescribed movement, primary joint, bilateral
  category, laterality) is loaded from the Thrive clinical database.

---

## 3. Per-rep kinematic metrics

All metrics are computed by `evaluation/metric_calculator.py` on the smoothed
angle time-series of the primary joint for each rep. Questions are built from
these metrics, and distractors are perturbations of them.

### 3.1 Angle extraction

- **Landmark triplet**: each joint angle is computed from an anatomical triplet
  (e.g. shoulder–hip–knee for hip flexion, hip–knee–ankle for knee flexion).
  Triplets are defined per joint in `utils/geometry.py`.
- **Signed 2D angle**: computed as the interior angle at the vertex landmark
  in the image plane, in degrees ∈ [0, 180].
- **Gaussian smoothing**: the per-frame angle series is smoothed with a
  Gaussian kernel (σ = 2 frames by default). Missing frames (no detection)
  are filled by neighbour interpolation before smoothing; long leading gaps
  are left as `None` to avoid spurious start values.
- **Fallback chain** for missing frames: within-rep interpolation → trailing
  backward fill → leave as None if still absent.

### 3.2 Rep-level metrics

Computed in `compute_rom_for_rep()`. A rep is skipped entirely if fewer than
3 valid (non-None) smoothed frames survive.

| Metric | Formula / definition | Units |
|---|---|---|
| **ROM** (`rom_degrees`) | Maximum absolute deviation of the angle from its starting value (`max abs(angle_t − angle_start)`) across the rep — the **maximum excursion from the starting position**. This differs from `max − min`; it is the metric that matches what clinicians observe ("how far did the patient move from their start position"). | degrees |
| **Peak angle** (`peak_angle_degrees`) | Angle at the frame of maximum excursion. For flexion exercises this is the mathematical minimum; for extension it is the maximum. Ties are resolved by taking the **last** frame, so holds report the end of the plateau. | degrees |
| **Min angle** (`min_angle_degrees`) | Mathematical minimum of the angle series. | degrees |
| **Mean angle** (`mean_angle_degrees`) | Arithmetic mean over all valid frames. | degrees |
| **Start / end angle** | First / last valid frame's angle. Used for direction inference. | degrees |
| **Peak frame** (`peak_frame`) | Frame index of peak excursion. | index |
| **Num frames** (`num_frames`) | Total frame count for the rep (including None frames). | count |
| **Peak ratio** | `peak_frame / num_frames` ∈ [0, 1]. Phase classification: **< 0.35 = early**, **0.35–0.65 = midpoint**, **> 0.65 = late**. Used by `tier_a_phase_identification`. | ratio |
| **Concentric velocity** (`concentric_velocity`) | `abs(peak − start) / (peak_frame − 0)` | degrees/frame |
| **Eccentric velocity** (`eccentric_velocity`) | `abs(peak − end) / (num_frames − peak_frame)` | degrees/frame |
| **Velocity ratio** (`velocity_ratio`) | `eccentric_velocity / concentric_velocity`. Controlled eccentric phase is a quality marker in rehab. `-1` if concentric velocity is near zero (undefined). | ratio |
| **Hold frames** (`hold_frames`) | Longest contiguous run of frames where frame-to-frame angle change is at most 1° (i.e. `abs(angle_t − angle_{t-1}) ≤ 1°`). Guard: only counted if `max_excursion > 5°`; else set to 0 (pure-transition reps). | count |
| **Hold time** (`hold_time_seconds`) | `hold_frames / fps`. | seconds |
| **FPS** | Video capture frame rate, read from `fps.txt` in the session dir. | Hz |

### 3.3 Video-level aggregates

Computed across the per-rep metrics for one video.

| Metric | Formula | Notes |
|---|---|---|
| **CoV** (`coefficient_of_variation`) | `(sample_std(rom_values) / mean(rom_values)) * 100` | Uses Bessel's correction (n−1). Returns `None` if mean < 1° (ROMs below 1° are noise-level). Values < ~15% = very consistent; > 30% = erratic. |
| **ROM trend** (`rom_trend`) | Classification from linear regression slope on per-rep ROM. Outliers trimmed via MAD (`σ ≈ 1.4826 * MAD`, reject beyond 2.5σ). Labels: **increasing** / **decreasing** / **stable**. | Requires slope magnitude ≥ 2 deg/rep **and** total predicted change (slope × (n−1)) ≥ 15°. Prevents labeling small absolute changes in short sets as trends. |
| **ROM slope** | Raw slope from the trimmed regression. | degrees/rep |
| **Active side per rep** | For alternating-bilateral exercises: the side (`left`/`right`/`both`) where the rep's ROM ≥ `ACTIVE_SIDE_ROM_THRESHOLD` (20°) AND exceeds the opposite side's ROM. | Used to label rep images in bilateral questions. |
| **Is alternating** | True if `active_side` alternates L/R/L/R in ≥ `ALTERNATION_THRESHOLD` (60%) of reps. | Drives Tier D gating. |

### 3.4 Compensatory motion detection

Compensatory = unwanted motion at a **non-target joint** that co-occurs with
the prescribed movement (e.g. lumbar flexion during hip abduction = spine
compensating for limited hip mobility).

- **Detection rule**: ROM on the compensatory joint exceeds the primary
  joint's ROM × some ratio, AND the peak-frames of the two joints are within
  N frames of each other (temporal correlation).
- **Output fields**: `compensatory_detected` (bool), `compensatory_joint`
  (joint name).
- **Eligible pairs** are defined per exercise in the metadata CSV
  (`compensatory_joint` column). Not all exercises have a defined
  compensatory pattern.

### 3.5 Numerical thresholds

These constants govern the entire dataset. Tuning any of them changes what
samples get generated.

| Constant | Value | Purpose |
|---|---|---|
| `MIN_ROM_FOR_QUESTIONS` | 10° | Skip joints with mean ROM below this for trend/CoV/phase questions. |
| `MIN_ROM_FOR_ASYMMETRY` | 10° | Skip L/R asymmetry questions when both sides are below this (noise floor). |
| `MAX_COV_FOR_VARIABILITY` | 80 % | Skip variability/consistency questions when CoV exceeds this — reps too erratic. |
| `MIN_ROM_DIFF_FOR_DIRECTION` | 2° | Below this absolute ROM difference, comparison says "similar" not "increased/decreased". |
| `MIN_ROM_DIFF_PCT_FOR_DIRECTION` | 15 % | Below this relative ROM difference, comparison says "similar" (catches 8% on large ROMs). |
| `ACTIVE_SIDE_ROM_THRESHOLD` | 20° | ROM above this = active side in bilateral rep labelling. |
| `ALTERNATION_THRESHOLD` | 60 % | Fraction of reps with clear L/R imbalance needed to classify a video as alternating. |
| Hold velocity threshold | 1°/frame | Frame-to-frame angle change below this counts as "static" for hold-frame measurement. |
| Hold excursion guard | 5° | Rep must have > 5° total excursion for hold measurement to activate (suppresses pure-transition reps). |
| Gaussian smoothing σ | 2 frames | Temporal smoothing kernel width. |
| Minimum valid frames / rep | 3 | Reps with < 3 non-None smoothed frames are skipped entirely. |

---

## 4. Question templates (tiers)

Questions are organized into four tiers of increasing reasoning complexity:

### Tier A — single-rep facts (~30% of dataset)
Questions about one specific rep.

| Template | What it asks |
|---|---|
| `tier_a_rom_single_rep` | "What is the ROM of rep N?" |
| `tier_a_peak_angle` | "What is the peak angle reached in rep N?" |
| `tier_a_phase_identification` | "In which phase (early/mid/late) does the peak occur?" |
| `tier_a_temporal_grounding` | "Which frames contain rep N?" (index-based temporal grounding) |
| `tier_a_hold_duration` | "How long does the patient hold the position?" (Hold-type exercises only) |

### Tier B — two-rep comparison (~50%)
Questions comparing two specific reps in the same video.

| Template | What it asks |
|---|---|
| `tier_b_rom_comparison` | "How does rep A's ROM compare to rep B's?" |
| `tier_b_peak_comparison` | "How does rep A's peak angle compare to rep B's?" |
| `tier_b_correctness_criteria` | "Which rep satisfies the prescribed execution criteria?" |
| `tier_b_compensatory` | "Which rep shows compensatory motion?" |

### Tier C — multi-rep trends (~20%)
Questions across the full rep sequence in one video.

| Template | What it asks |
|---|---|
| `tier_c_kinematic_trend` | "Does ROM increase/decrease/stay flat across reps?" |
| `tier_c_variability_analysis` | "Which joint shows the most consistent performance?" |

### Tier D — bilateral symmetry (~1–2%)
Only applicable to `alternating_bilateral` exercises (e.g. alternating lunges).

| Template | What it asks |
|---|---|
| `tier_d_side_consistency` | "Is the movement symmetric between sides?" |

---

## 5. Filters that gate question generation

Raw-metric eligibility is not enough — many reps are unreliable for clinical
reasoning. The generator applies filters at multiple layers:

### Rep-level filters
- **Minimum ROM**: reps with ROM below the exercise-specific `min_rom_target`
  are dropped. No meaningful question can be built on a rep where the patient
  barely moved.
- **Truncation detection**: reps whose motion curve is cut off at the video
  boundary (peak in first or last frame) are dropped.
- **Contiguous-window preservation**: for continuous-motion templates, reps
  are kept as a contiguous window — isolated "good" reps between "bad" reps
  are dropped so the model sees an uninterrupted motion sequence. Small gaps
  (≤ 2 bad reps) inside an otherwise-good window are tolerated.

### Video-level filters
- **`excluded_exercise_codes`**: 12 thumb/finger/hand-tendon codes (15000–15011)
  are fully excluded. BlazePose produced empty keypoints for these in the
  upstream pipeline, and VitPose cannot resolve fingertip motion. Documented
  in `EXCLUDED_EXERCISES.md`.
- **`excluded_videos`**: per-video manual blocklist for known-bad sessions.
- **Missing source frames**: videos lacking both `cropped_images/` and
  `images/` on disk produce no samples (nothing renderable).

### Template-level filters
- **`eligible_templates` per exercise**: exercises with poor 2D coverage
  (e.g. sidelying frontal-plane movement) are restricted to phase/temporal
  templates only. ROM-based templates on these exercises would measure
  out-of-plane projections, which are clinically meaningless.
- **`vitpose_swap`**: camera-perspective L/R labels are swapped for all
  front-facing VitPose exercises via the keypoint adapter (see
  `REPORT_SIDE_SWAP.md`). Questions about a specific side are only generated
  once the swap is applied.
- **ROM safety net**: if a question is about one side but the opposite side
  shows ≥ 15° more ROM (and ≥ 2× ratio), the question is dropped — likely a
  side-labelling error the swap missed.
- **Tier B block for Hold exercises**: `tier_b_rom_comparison` and
  `tier_b_peak_comparison` are structurally invalid for static holds and are
  blocked explicitly.
- **Tier D gating**: Tier D is gated to `alternating_bilateral` exercises
  only. Simultaneous-bilateral exercises produce meaningless side-symmetry
  questions when both sides move in lockstep.
- **CoV cap**: Tier C variability questions are only generated when CoV <
  80%. Above that the signal is too noisy to claim a meaningful consistency
  finding.

---

## 6. Distractors

Distractors are drawn from a fixed error taxonomy specific to each template.
Example for `tier_b_rom_comparison`:

| Error class | What the distractor does |
|---|---|
| `magnitude_swap` | Correct direction, wrong magnitude |
| `direction_flip` | Wrong direction (increase vs decrease) |
| `no_difference` | Claims reps are similar when they differ |
| `wrong_side` | Correct pattern on the wrong side |

Each template has 3–5 error classes; 3 are sampled per question. This keeps
distractors realistic (they look like clinical mistakes) and the task
clinically meaningful (a random-letter baseline is defeated by picking any
measurable signal).

---

## 7. Quality controls

- **Deduplication**: near-duplicate questions within a video (Jaccard on the
  question stem) are dropped.
- **Per-video question cap**: at most `max_per_video` (default 6) questions
  per video, to prevent the same video from dominating the dataset.
- **Per-template cap**: at most `max_per_template` (default 1) per video, for
  template diversity.
- **Train/test video-level split**: no video appears in both splits. Samples
  are grouped by `video_id` first, then splits are assigned.
- **Metadata `verification` block**: every sample carries the raw metrics
  used to compute the answer. This is how we audit correctness at scale
  (any downstream check can re-verify without re-running the pipeline).

---

## 8. Per-sample metadata (what survives to training)

| Field | Meaning |
|---|---|
| `video_id`, `exercise_code`, `exercise_name` | Identity |
| `body_region` | `upper_extremity` / `lower_extremity` / `hands` / `feet` / `whole_body` / `spine_core` |
| `bilateral_category` | `unilateral` / `alternating_bilateral` / `simultaneous_bilateral` / `within_rep_bilateral` |
| `keypoint_source` | `vitpose` or `blazepose` |
| `exercise_type` | `Reps` / `Hold` / `Steps` |
| `frames_source` | `cropped` (standard) or `uncropped` (fallback when upstream cropping missing) |
| `difficulty_tier` | `TIER_A` / `TIER_B` / `TIER_C` / `TIER_D` |
| `question_template` | e.g. `tier_b_rom_comparison` |
| `verification` | raw metric values (ROM, peak_ratio, CoV, computation_method, joint, etc.) |
| `rep_comparison` | (Tier B only) `{rep_a, rep_b, rep_a_label, rep_b_label}` |
| `active_side_per_rep` | (alternating-bilateral only) which side each rep was performed on |

---

## 9. End-to-end pipeline

1. **Generate** — `qa_generator.py` produces raw JSONL with multimodal
   `messages` + `metadata`.
2. **Split** — `split_dataset.py` produces train/val/test JSONL grouped by
   `video_id`.
3. **Prepare** — `prepare_video_inputs.py` converts multimodal content to
   Thrive-schema (`video_frames`, stringified `messages`, `fps`,
   `num_frames`, `need_to_flip`).
4. **Package** — `package_to_hf.py` produces an HF `DatasetDict`.
5. **Build final** — `native_frame_rebuild/build_mcqa_video_0804.py`
   produces the final dataset with both HF Arrow splits and OpenAI SFT
   JSONL (`openai_sft/{train,validation,test}.jsonl`).

---

## 10. Known limitations

- **Saturation risk**: since train and test share exercise families and
  prompt templates, high MCQA accuracy can partly reflect text-pattern
  learning rather than robust video understanding.
- **Camera-perspective labels**: upstream pose data uses image-relative L/R,
  which is flipped vs anatomical L/R for front-facing exercises. Handled via
  the VitPose-swap flag, but sidelying and mixed-view exercises may still
  have edge cases.
- **BlazePose hand gaps**: fingertip-tracking data is missing for 12 codes
  (15000–15011), which is why those are excluded.
"""


def build_ui():
    with gr.Blocks(title="Video SFT Dataset Monitor") as app:
        gr.Markdown("# Video SFT Dataset Monitor")

        # --- State ---
        samples_state = gr.State([])
        index_state = gr.State({})
        filtered_state = gr.State([])
        pos_state = gr.State(0)
        annotations_state = gr.State({})
        annotations_path_state = gr.State("")
        reasoning_index_state = gr.State({})

        with gr.Row():
            # ============ SIDEBAR ============
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### Controls")

                _browse_datasets = _scan_browse_datasets()
                _browse_choices = [label for label, _ in _browse_datasets]
                _browse_paths = {label: path for label, path in _browse_datasets}
                dataset_picker = gr.Dropdown(
                    label="Quick-load dataset",
                    choices=_browse_choices,
                    value=None,
                    interactive=True,
                    info="Scans app_video_datasets/ — selecting loads immediately",
                )

                jsonl_input = gr.Textbox(
                    label="JSONL Path", value=CONFIG["default_jsonl"],
                    lines=1, max_lines=1,
                )
                load_btn = gr.Button("Load JSONL", variant="primary")

                # Dataset filter sits directly under Load JSONL — it's a
                # top-level concern (which dataset within a combined file).
                # Only meaningful for a combined file (e.g.
                # questions_3d_v3_plus_v21.jsonl); collapses to ["All"] on a
                # single-dataset file.
                source_dataset_filter = gr.Dropdown(
                    label="Dataset",
                    choices=["All"], value="All", interactive=True,
                )

                v6_jsonl_input = gr.Textbox(
                    label="Comparison JSONL (other version, optional)",
                    value="/mnt/data/sgsilva/tmp/qa_all_exercises_v5.jsonl",
                    lines=1, max_lines=1,
                )
                v6_load_btn = gr.Button("Load comparison JSONL", size="sm")
                v6_load_status = gr.Markdown("*No comparison loaded*")

                sample_counter = gr.Markdown("*No samples loaded*")

                with gr.Row():
                    prev_btn = gr.Button("◄ Prev", size="sm")
                    next_btn = gr.Button("Next ►", size="sm")
                random_btn = gr.Button("🎲 Random", size="sm")
                refresh_btn = gr.Button("🔄 Refresh", size="sm")
                clear_filters_btn = gr.Button("✖ Clear filters", size="sm")

                with gr.Row():
                    video_search = gr.Textbox(
                        label="Search Video ID", placeholder="e.g. 10025",
                        lines=1, max_lines=1, scale=3,
                    )
                    video_search_btn = gr.Button("Go", size="sm", scale=0)

                tier_filter = gr.Dropdown(
                    label="Tier / Question Type", choices=["All"], value=[],
                    multiselect=True, interactive=True,
                )
                joint_filter = gr.Dropdown(
                    label="Joint", choices=["All"], value="All", interactive=True,
                )
                exercise_filter = gr.Dropdown(
                    label="Exercise", choices=["All"], value="All", interactive=True,
                )
                exercise_id_filter = gr.Dropdown(
                    label="Exercise ID", choices=["All"], value="All", interactive=True,
                )
                exercise_type_filter = gr.Dropdown(
                    label="Exercise Type", choices=["All"], value="All", interactive=True,
                )
                category_filter = gr.Dropdown(
                    label="Category", choices=["All"], value="All", interactive=True,
                )
                label_filter = gr.Dropdown(
                    label="Movement Label",
                    choices=["All", "correct", "incorrect", "mixed", "unknown"],
                    value="All", interactive=True,
                )
                body_region_filter = gr.Dropdown(
                    label="Body Region",
                    choices=["All"], value="All", interactive=True,
                )
                kp_source_filter = gr.Dropdown(
                    label="Keypoint Source",
                    choices=["All", "vitpose", "blazepose", "sam3dbody"],
                    value="All", interactive=True,
                )
                frames_source_filter = gr.Dropdown(
                    label="Frames Source",
                    choices=["All", "cropped", "uncropped"],
                    value="All", interactive=True,
                )
                camera_perspective_filter = gr.Dropdown(
                    label="Camera Perspective",
                    choices=["All", "frontal", "lateral"],
                    value="All", interactive=True,
                )
                annotation_filter = gr.Dropdown(
                    label="Annotation",
                    choices=["All", "Has issues", "Maybe redundant", "Good quality", "Unrated"],
                    value="All", interactive=True,
                )
                # Salvage origin: distinguishes samples carried over from prior
                # dataset versions (human-audited as Good quality, then passed
                # the v14_added filters) from net-new samples generated by the
                # current build. See scripts/build_human_salvage.py +
                # REPORT_SALVAGE_HUMAN.md. Surfaces metadata.salvaged_human.
                salvage_origin_filter = gr.Dropdown(
                    label="Salvage origin",
                    choices=[
                        "All",
                        "Salvaged (human Good quality, prior version)",
                        "Salvaged & needs reverification",
                        "Salvaged & confirmed by sgsilva",
                        "Net-new (current version)",
                    ],
                    value="All", interactive=True,
                )
                # Geometry-assessment: surfaces the abduction sagittal-chain audit.
                # MODE_2_DROP = wrong-chain abduction (calc invalid); MODE_1_KEEP = wrong
                # joint but valid sagittal-flexion calculation.
                geometry_assessment_filter = gr.Dropdown(
                    label="Geometry assessment",
                    choices=[
                        "All",
                        "MODE_2_DROP",
                        "MODE_1_KEEP",
                        "Any (has assessment)",
                        "Unassessed",
                    ],
                    value="All", interactive=True,
                )
                annotated_by_filter = gr.Dropdown(
                    label="Annotated by",
                    choices=["All"], value="All", interactive=True,
                )
                # Judge-verdict filter: surfaces the LLM-judge categories from
                # judge_flags_*.jsonl files. "Any non-ok" is a quick way to find
                # everything the judge flagged regardless of specific category.
                judge_pass1_filter = gr.Dropdown(
                    label="Pass 1 verdict (per-rep video audit)",
                    choices=[
                        "All",
                        "Any non-ok",
                        "No verdict",
                        "ok",
                        "patient_not_performing",
                        "multi_rep_in_one_rep",
                        "wrong_exercise",
                        "pose_estimation_wrong_subject",
                        "wrong_side_for_question",
                        "parse_failed",
                        "error",
                        "pending",
                    ],
                    value="All", interactive=True,
                )
                judge_pass2_filter = gr.Dropdown(
                    label="Pass 2 verdict (side-mismatch check)",
                    choices=[
                        "All",
                        "Any non-ok",
                        "No verdict",
                        "side_matches",
                        "side_mismatch",
                        "side_not_applicable",
                        "parse_failed",
                        "error",
                        "pending",
                    ],
                    value="All", interactive=True,
                )
                diff_status_filter = gr.Dropdown(
                    label="Diff vs Comparison",
                    choices=["All", "Changed", "Added", "Same"],
                    value="All", interactive=True,
                )
                # Note: "Removed" isn't selectable here — those slots only
                # exist in the comparison file (not in the primary sample list
                # we navigate). To inspect Removed slots, swap which file is
                # primary and reload.

                with gr.Accordion("Question Types (14)", open=False):
                    gr.Markdown(
                        "**Tier A — Single Rep**\n"
                        "1. `tier_a_rom_single_rep` — ROM fact (excluded for valley/flexion exercises)\n"
                        "2. `tier_a_peak_angle` — Peak angle at max excursion from start\n"
                        "3. `tier_a_phase_identification` — When does peak movement occur?\n"
                        "4. `tier_a_temporal_grounding` — Peak at what % of the rep?\n\n"
                        "**Tier B — Comparative (2-rep)**\n"
                        "5. `tier_b_rom_comparison` — Rep A vs Rep B ROM (excluded for valley exercises)\n"
                        "6. `tier_b_peak_comparison` — Rep A vs Rep B peak angle\n"
                        "7. `tier_b_correctness_criteria` — Exercise form criteria vs data\n"
                        "8. `tier_b_compensatory` — Non-primary joint compensatory movement\n\n"
                        "**Tier C — Multi-rep (full set)**\n"
                        "9. `tier_c_kinematic_trend` — ROM trend across reps\n"
                        "10. `tier_c_variability_analysis` — CoV / consistency analysis\n\n"
                        "**Tier D — Bilateral (bilateral exercises only)**\n"
                        "11. `tier_d_side_comparison` — Left vs right mean ROM\n"
                        "12. `tier_d_asymmetry_detection` — L/R asymmetry %\n"
                        "13. `tier_d_side_consistency` — L/R CoV comparison\n"
                        "14. `tier_d_peak_comparison` — L/R mean peak angle\n\n"
                        "*Valley/flexion exercises exclude ROM templates. Hold exercises: peak angle only. "
                        "Poor 2D (sidelying): phase/temporal/peak angle only. "
                        "Phase/correctness require exercise metadata.*"
                    )

                with gr.Accordion("Metrics Glossary", open=False):
                    gr.Markdown(
                        "<small>\n\n"
                        "| Field | Description |\n"
                        "|-------|-------------|\n"
                        "| **rep\\_index** | Rep number (1-based, filesystem index before edge trimming) |\n"
                        "| **joint\\_name** | Joint identifier (e.g. left\\_knee, right\\_hip) |\n"
                        "| **rom\\_degrees** | max(abs(angle\\_t − start\\_angle)) — maximum angular excursion from starting position. Direction-agnostic: correct for both flexion (angle decreases) and extension (angle increases). |\n"
                        "| **start\\_angle\\_degrees** | Angle at first valid frame — pre-movement baseline used as the excursion reference. |\n"
                        "| **peak\\_angle\\_degrees** | Absolute joint angle at the frame of maximum excursion from start (the clinical peak). For flexion exercises this is the mathematical minimum. |\n"
                        "| **peak\\_frame** | Frame index of maximum excursion — the clinical peak frame used in temporal grounding questions. |\n"
                        "| **peak\\_ratio** | peak\\_frame / num\\_frames (range 0.0–1.0) — fractional position of the peak within the rep's duration. Classified by `tier_a_phase_identification`: < 0.35 = early, 0.35–0.65 = midpoint, > 0.65 = late. |\n"
                        "| **min\\_angle\\_degrees** | Mathematical minimum angle across the rep (for reference and display). |\n"
                        "| **min\\_frame** | Frame index of the mathematical minimum angle. |\n"
                        "| **end\\_angle\\_degrees** | Angle at last valid frame — return-to-start position. Used in half-cycle detection. |\n"
                        "| **mean\\_angle\\_degrees** | Mean angle across all rep frames. Posture baseline reference. |\n"
                        "| **num\\_frames** | Total frames in the rep (including low-confidence frames). |\n"
                        "| **fps** | Frame rate of the recording (from fps.txt, typically ~8 Hz). Used to convert frame counts to seconds. |\n"
                        "| **concentric\\_velocity** | abs(peak − start) / concentric\\_frames — angular speed from start to peak (deg/frame). Measures effort in the active phase. |\n"
                        "| **eccentric\\_velocity** | abs(peak − end) / eccentric\\_frames — angular speed from peak to end (deg/frame). Measures control in the return phase. |\n"
                        "| **velocity\\_ratio** | eccentric\\_vel / concentric\\_vel. Ratio > 2.0 indicates poor eccentric control; −1 when concentric velocity is near zero. |\n"
                        "| **hold\\_frames** | Frames within 5% of peak excursion magnitude — time spent at end range. Only computed when ROM > 5°. |\n"
                        "| **hold\\_time\\_seconds** | hold\\_frames / fps — seconds spent at end range. |\n"
                        "| **active\\_side** | For bilateral exercises: left/right — which side is the working limb for this rep. Derived from which side has larger ROM. |\n"
                        "| **CoV** | (sample\\_std / mean) × 100% — Coefficient of Variation across reps. < 15% = consistent, 15–30% = moderate, > 30% = variable. Uses Bessel's correction (n−1). |\n"
                        "| **Trend** | ROM trend across reps via linear regression slope. Classified as *increasing*, *decreasing*, or *stable* (requires slope > 2°/rep and total change > 15°). |\n"
                        "| **L/R ROM diff** | abs(left\\_ROM − right\\_ROM) / mean(left\\_ROM, right\\_ROM) × 100% — bilateral asymmetry. < 20% = symmetrical. |\n\n"
                        "</small>"
                    )

                with gr.Accordion("Distractor Types", open=False):
                    gr.Markdown(
                        "| Type | Strategy |\n"
                        "|------|----------|\n"
                        "| `reversed_comparison` | Swaps which rep has higher/lower ROM |\n"
                        "| `error_misattribution` | Correct values, wrong clinical error label |\n"
                        "| `magnitude_shifted` | Correct direction, wrong ROM magnitude |\n"
                        "| `wrong_rom_value` | Correct joint/rep, shifted ROM (30-50%) |\n"
                        "| `wrong_joint` | Values attributed to a different joint |\n"
                        "| `reversed_trend` | Correct mean/CoV, opposite trend direction |\n"
                        "| `wrong_consistency` | Correct trend, inflated/deflated CoV |\n"
                        "| `wrong_error` | Claims a different error than detected |\n"
                        "| `no_errors` | Claims no errors when one is present |\n"
                        "| `wrong_timestamp` | Peak frame shifted by ~50% |\n"
                        "| `start_of_rep` | Claims peak occurs at frame 0 |\n"
                        "| `wrong_phase` | Reversed timing (concentric vs eccentric) |\n"
                        "| `no_peak` | Claims no distinct peak exists |\n"
                        "| `criterion_not_met` | Claims a criterion is not met when it is |\n"
                    )

            # ============ MAIN CONTENT ============
            with gr.Column(scale=3):
                with gr.Tabs():
                    # ---- TAB 1: MCQA Browser ----
                    with gr.Tab("MCQA Browser"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                with gr.Tabs():
                                    with gr.Tab("Video"):
                                        mcqa_video = gr.Video(label="Exercise Video", show_label=False, height=350)
                                    with gr.Tab("Frames"):
                                        # Scrollable gallery — fixed 4 columns, height capped, rows
                                        # grow with frame count up to max_gallery_frames (32).
                                        mcqa_gallery = gr.Gallery(
                                            label="Frame Gallery", columns=4,
                                            height=600, object_fit="contain",
                                            allow_preview=True,
                                        )
                                mcqa_rep_info = gr.Markdown(
                                    "*Select a sample*",
                                    min_height=60,
                                )
                                with gr.Row():
                                    mcqa_rep_selector = gr.Dropdown(
                                        label="Rep", choices=[], value=None,
                                        interactive=True, scale=0, min_width=80,
                                        allow_custom_value=True,
                                    )
                                    mcqa_allreps_cb = gr.Checkbox(
                                        label="All Reps", value=False, scale=0,
                                    )
                                    mcqa_skeleton_cb = gr.Checkbox(
                                        label="Skeleton", value=False, scale=0,
                                    )

                            with gr.Column(scale=1):
                                mcqa_question = gr.Markdown("*Load JSONL to browse questions*")

                        with gr.Row():
                            with gr.Column():
                                mcqa_verification = gr.Markdown("")
                            with gr.Column():
                                mcqa_distractors = gr.Markdown("")

                        with gr.Accordion("Exercise Info", open=False):
                            mcqa_exercise_detail = gr.Markdown("*No exercise info*")

                        with gr.Accordion("Full Metrics", open=False):
                            mcqa_metrics_md = gr.Markdown("*Select a sample to compute metrics*")

                        with gr.Accordion("Reasoning", open=False):
                            mcqa_reasoning_md = gr.Markdown("*Select a sample*")

                        with gr.Accordion("Video Metadata", open=False):
                            mcqa_video_meta = gr.Markdown("*No video selected*")

                        with gr.Accordion("Full QA Metadata", open=False):
                            mcqa_metadata_json = gr.JSON(label="Raw Metadata")

                        with gr.Accordion("Side-by-side comparison", open=True):
                            v6_compare_md = gr.Markdown("*Load a comparison JSONL to enable side-by-side comparison*")

                        with gr.Accordion("Quality Audit", open=True):
                            # Prominent display of any existing annotation (rating + note)
                            audit_status = gr.Markdown("")
                            gr.Markdown("**Rate this question's quality:**")
                            audit_rating = gr.Radio(
                                choices=["Good quality", "Maybe redundant", "Has issues"],
                                value=None, label="Rating", interactive=True,
                            )
                            audit_note = gr.Textbox(
                                label="Notes (required for 'Has issues')",
                                placeholder="Describe the issue...",
                                lines=2, max_lines=4, interactive=True,
                            )
                            with gr.Row():
                                audit_save_btn = gr.Button("Save Annotation", variant="primary", size="sm")
                                exclude_btn = gr.Button("🚫 Exclude question", variant="stop", size="sm")
                            exclude_status = gr.Markdown("")
                            audit_counter = gr.Markdown("*No annotations yet*")

                        with gr.Accordion("Judge Verdict (LLM)", open=False):
                            mcqa_judge_md = gr.Markdown("*No judge verdict for any rep referenced by this sample*")

                        with gr.Accordion("Statistics", open=False):
                            with gr.Row():
                                tier_chart = gr.Plot(label="Tiers")
                                joint_chart = gr.Plot(label="Joints")
                            distractor_chart = gr.Plot(label="Distractors")
                            with gr.Row():
                                ex_type_template_chart = gr.Plot(label="Templates per Exercise Type")
                                ex_type_tier_chart = gr.Plot(label="Tiers per Exercise Type")

                    # ---- TAB 2: Exercise Explorer ----
                    with gr.Tab("Exercise Explorer", visible=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                ex_region_filter = gr.Dropdown(
                                    label="Body Region",
                                    choices=["All", "lower_body", "upper_body", "resistance"],
                                    value="All", interactive=True,
                                )
                                ex_search = gr.Textbox(
                                    label="Search Exercise", placeholder="e.g. Bridge",
                                    lines=1, max_lines=1,
                                )
                                ex_table = gr.Dataframe(
                                    headers=["Code", "Exercise", "Region", "Videos"],
                                    datatype=["str", "str", "str", "number"],
                                    interactive=False,
                                    label="Exercises",
                                )

                            with gr.Column(scale=2):
                                # Video player + frames at top
                                with gr.Row():
                                    with gr.Column():
                                        ex_video_player = gr.Video(label="Video", height=300)
                                    with gr.Column():
                                        ex_gallery = gr.Gallery(
                                            label="Frames", columns=4, rows=2,
                                            height=300, object_fit="contain",
                                        )
                                with gr.Row():
                                    ex_rep_dropdown = gr.Dropdown(
                                        label="Repetition", choices=[], value=None,
                                        interactive=True, scale=1,
                                    )
                                    ex_cropped_cb = gr.Checkbox(
                                        label="Cropped", value=False, scale=0,
                                    )
                                    ex_skeleton_cb = gr.Checkbox(
                                        label="Skeleton", value=False, scale=0,
                                    )
                                    ex_video_id_state = gr.State("")
                                    ex_exercise_code_state = gr.State("")
                                with gr.Accordion("Video Metadata", open=False):
                                    ex_video_meta = gr.Markdown("*No video selected*")

                                gr.Markdown("---")

                                # Video table with label filter
                                with gr.Row():
                                    ex_videos_label = gr.Markdown("")
                                    ex_label_filter = gr.Dropdown(
                                        label="Movement Label",
                                        choices=["All", "correct", "incorrect", "mixed", "unknown"],
                                        value="All", interactive=True, scale=0,
                                    )
                                ex_video_table = gr.Dataframe(
                                    headers=["Video ID", "Reps", "Label", "Exercise ID"],
                                    datatype=["str", "number", "str", "str"],
                                    interactive=False, max_height=200,
                                    label="Matching Videos",
                                )

                                # Exercise detail below
                                ex_detail = gr.Markdown("*Select an exercise from the table*")

                    # ---- Placeholder tabs ----
                    with gr.Tab("Metrics Dashboard", visible=False):
                        gr.Markdown("*Coming soon — ROM per rep charts, CoV visualization, trend analysis*")

                    with gr.Tab("VLM Analysis", visible=False):
                        gr.Markdown("*Coming soon — Live VLM query interface*")

                    with gr.Tab("Knowledge Graph", visible=False):
                        gr.Markdown("### Physiotherapy Knowledge Graph (PKG)")
                        with gr.Row():
                            with gr.Column(scale=1):
                                pkg_exercise_dd = gr.Dropdown(
                                    label="Filter by Exercise",
                                    choices=[("(all)", "(all)")],
                                    value="(all)",
                                    allow_custom_value=True,
                                )
                                pkg_muscle_dd = gr.Dropdown(
                                    label="Filter by Muscle",
                                    choices=[("(all)", "(all)")],
                                    value="(all)",
                                    allow_custom_value=True,
                                )
                                pkg_region_dd = gr.Dropdown(
                                    label="Filter by Body Region",
                                    choices=[("(all)", "(all)"), ("Lower Body", "lower_body"), ("Upper Body", "upper_body"), ("Resistance", "resistance")],
                                    value="(all)",
                                )
                                pkg_show_similar = gr.Checkbox(
                                    label="Show SIMILAR_TO edges",
                                    value=True,
                                )
                                pkg_show_phases = gr.Checkbox(
                                    label="Show phases",
                                    value=False,
                                )
                                pkg_refresh_btn = gr.Button("Refresh Graph", variant="primary")
                                pkg_status = gr.Markdown("*Load the graph to explore*")
                                # Hidden textbox for double-click navigation from iframe
                                pkg_nav_target = gr.Textbox(visible=False, elem_id="pkg_nav_target")
                            with gr.Column(scale=4):
                                pkg_html = gr.HTML(label="Interactive Graph")
                                # JS listener: iframe postMessage → hidden textbox
                                gr.HTML("""
                                <script>
                                window.addEventListener("message", function(event) {
                                    if (event.data && event.data.type === "pkg_navigate") {
                                        var el = document.querySelector("#pkg_nav_target textarea");
                                        if (el) {
                                            var nativeSet = Object.getOwnPropertyDescriptor(
                                                window.HTMLTextAreaElement.prototype, "value").set;
                                            nativeSet.call(el, event.data.exercise_code);
                                            el.dispatchEvent(new Event("input", {bubbles: true}));
                                        }
                                    }
                                });
                                </script>
                                """)

        # ==================================================================
        # 10. Event handlers
        # ==================================================================

        # ---- PKG: Knowledge Graph ----
        _pkg_graph = None
        _pkg_graph_path = PKG_GRAPH_PATH

        def _load_pkg():
            nonlocal _pkg_graph
            if _pkg_graph is None:
                try:
                    from build_pkg import load_graph as _lg
                    _pkg_graph = _lg(_pkg_graph_path)
                except Exception as e:
                    logger.warning(f"PKG load failed: {e}")
                    return None
            return _pkg_graph

        def _populate_pkg_dropdowns():
            G = _load_pkg()
            if G is None:
                return [("(all)", "(all)")], [("(all)", "(all)")]
            exercises = [("(all)", "(all)")] + sorted(
                (f"{d['name']} ({n.split(':')[1]})", n.split(":")[1])
                for n, d in G.nodes(data=True)
                if d.get("type") == "exercise"
            )
            muscles = [("(all)", "(all)")] + sorted(
                (d.get("canonical_name", "").replace("_", " ").title(),
                 d.get("canonical_name", ""))
                for n, d in G.nodes(data=True)
                if d.get("type") == "muscle"
            )
            return exercises, muscles

        def on_pkg_refresh(exercise_val, muscle_val, region_val, show_similar, show_phases):
            G = _load_pkg()
            if G is None:
                return "<p>Graph not found. Run: <code>python scripts/build_pkg.py</code></p>", "*Not loaded*"
            try:
                from pkg_visualizer import generate_html, _filter_subgraph

                filt_ex = exercise_val if exercise_val and exercise_val != "(all)" else None
                filt_mu = muscle_val if muscle_val and muscle_val != "(all)" else None
                filt_rg = region_val if region_val and region_val != "(all)" else None

                raw_html = generate_html(
                    G, filter_exercise=filt_ex, filter_muscle=filt_mu,
                    filter_body_region=filt_rg, show_similar_to=show_similar,
                    show_phases=show_phases,
                )
                sub = _filter_subgraph(
                    G, filter_exercise=filt_ex, filter_muscle=filt_mu,
                    filter_body_region=filt_rg, show_similar_to=show_similar,
                )
                import base64
                b64 = base64.b64encode(raw_html.encode("utf-8")).decode("utf-8")
                iframe = (
                    f'<iframe src="data:text/html;base64,{b64}" '
                    f'width="100%" height="800px" style="border:none;"></iframe>'
                )
                return iframe, f"**{sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges**"
            except Exception as e:
                logger.warning(f"PKG render failed: {e}")
                return f"<p>Error: {e}</p>", f"*Error: {e}*"

        pkg_refresh_btn.click(
            fn=on_pkg_refresh,
            inputs=[pkg_exercise_dd, pkg_muscle_dd, pkg_region_dd, pkg_show_similar, pkg_show_phases],
            outputs=[pkg_html, pkg_status],
        )

        def on_pkg_navigate(exercise_code, show_similar, show_phases):
            """Handle double-click navigation: focus on the clicked exercise."""
            if not exercise_code or not exercise_code.strip():
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            code = exercise_code.strip()
            # Render the focused exercise graph
            html, status = on_pkg_refresh(code, None, None, show_similar, show_phases)
            # Update exercise dropdown to match, clear other filters
            return html, status, code, "(all)", "(all)"

        pkg_nav_target.change(
            fn=on_pkg_navigate,
            inputs=[pkg_nav_target, pkg_show_similar, pkg_show_phases],
            outputs=[pkg_html, pkg_status, pkg_exercise_dd, pkg_muscle_dd, pkg_region_dd],
        )

        try:
            _ex_choices, _mu_choices = _populate_pkg_dropdowns()
            pkg_exercise_dd.choices = _ex_choices
            pkg_muscle_dd.choices = _mu_choices
        except Exception:
            pass

        # ---- MCQA: Load ----
        def on_load(jsonl_path, skeleton, all_reps):
            try:
                _redirected = _redirect_features_to_questions(jsonl_path)
                if _redirected != jsonl_path:
                    gr.Info(f"Loaded the questions file {Path(_redirected).name} "
                            f"(you pointed at a features sidecar).")
                    jsonl_path = _redirected
                samples = load_jsonl_samples(jsonl_path)
            except Exception as e:
                gr.Warning(f"Failed to load: {e}")
                # Must match outputs: 7 states + counter + 10 filters + 14 render + 5 charts + ann_counter
                # Filters: tier(list), joint, exercise, body_region, kp_source,
                # source_dataset, exercise_id, exercise_type, category, annotated_by
                return ([], {}, [], 0, {}, "", {},
                        "*Error loading file*", [], "All", "All", "All", "All", "All", "All", "All", "All", "All",
                        "*Error*", None, [], "*No rep info*", gr.update(choices=[], value=None),
                        "", "", "*No exercise info*", "*No metrics*",
                        "*No video*", {},
                        None, "", "", "*No reasoning*",
                        "*No judge verdict for any rep referenced by this sample*",
                        empty_figure(), empty_figure(), empty_figure(),
                        empty_figure(), empty_figure(),
                        "*No annotations yet*")

            index = build_index(samples)
            filtered = list(range(len(samples)))
            annotations = _load_annotations(jsonl_path)
            reasoning_index = load_reasoning_index(jsonl_path)
            _load_video_metrics_sidecar(jsonl_path)
            # Invalidate the per-video metrics cache so the new sidecar is used.
            if hasattr(get_video_metrics_cached, "_cache"):
                get_video_metrics_cached._cache.clear()

            # Readable labels for tiers
            _tier_labels = {
                "TIER_A_SINGLE_REP": "Tier A (single rep)",
                "TIER_B_COMPARATIVE": "Tier B (comparative)",
                "TIER_C_LONGITUDINAL": "Tier C (multi-rep)",
                "TIER_D_BILATERAL": "Tier D (bilateral)",
            }
            tier_choices = sorted(
                [_tier_labels.get(t, t) for t in index["tiers"]]
                + list(index["templates"].keys()),
                key=str.lower,
            )
            joint_choices = ["All"] + sorted(index["joints"].keys())
            exercise_choices = ["All"] + sorted(index["exercises"].keys())
            # Exercise IDs sorted numerically
            exercise_id_choices = ["All"] + sorted(
                index["exercise_ids"].keys(), key=lambda x: int(x) if x.isdigit() else x
            )
            exercise_type_choices = ["All"] + sorted(index["exercise_types"].keys())
            category_choices = ["All"] + sorted(index["categories"].keys())
            body_region_choices = ["All"] + sorted(index["body_regions"].keys())
            kp_source_choices = ["All"] + sorted(index["kp_sources"].keys())
            # Dataset filter: only meaningful for a combined file. When the
            # only bucket is "(single)" the dropdown collapses to ["All"].
            _ds_keys = [k for k in index["source_datasets"].keys() if k != "(single)"]
            source_dataset_choices = ["All"] + sorted(_ds_keys)
            # Distinct annotation authors (for "Annotated by" dropdown). Includes
            # "(unknown)" bucket for legacy entries that predate author-tagging.
            _authors: set = set()
            for _ann in (annotations or {}).values():
                _authors.add(_ann.get("author") or "(unknown)")
            annotated_by_choices = ["All"] + sorted(_authors)

            # Render first sample (guarded so filter dropdowns always populate)
            try:
                result = render_sample(samples, filtered, 0, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=index, reasoning_index=reasoning_index)
            except Exception as e:
                logger.warning(f"render_sample failed on first sample: {e}")
                result = ("*Error rendering first sample*",
                          None, [], "*No rep info*", gr.update(choices=[], value=None),
                          "", "", "*No exercise info*", "*No metrics*",
                          "*No video*", {},
                          None, "", "", "*No reasoning*",
                          "*No judge verdict for any rep referenced by this sample*")

            return (
                samples, index, filtered, 0, annotations, jsonl_path, reasoning_index,  # states
                f"**Sample 1 / {len(filtered)}**",
                gr.update(choices=tier_choices, value=[]),
                gr.update(choices=joint_choices, value="All"),
                gr.update(choices=exercise_choices, value="All"),
                gr.update(choices=body_region_choices, value="All"),
                gr.update(choices=kp_source_choices, value="All"),
                gr.update(choices=source_dataset_choices, value="All"),
                gr.update(choices=exercise_id_choices, value="All"),
                gr.update(choices=exercise_type_choices, value="All"),
                gr.update(choices=category_choices, value="All"),
                gr.update(choices=annotated_by_choices, value="All"),
                *result,
                create_tier_chart(samples),
                create_joint_chart(samples),
                create_distractor_chart(samples),
                create_exercise_type_chart(samples),
                create_tier_per_exercise_type_chart(samples),
                _annotation_counter_md(annotations),
            )

        def _get_available_reps(video_id: str) -> List[str]:
            """Discover available rep indices for a video."""
            video_dir = resolve_video_dir(video_id)
            reps_dir = video_dir / "repetitions"
            reps = []
            if reps_dir.exists():
                for d in sorted(reps_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("repetition_"):
                        try:
                            idx = int(d.name.split("_")[-1])
                            reps.append(str(idx))
                        except (ValueError, IndexError):
                            pass
            return reps

        def render_sample(samples, filtered, pos, skeleton=False, all_reps=False, override_rep=None, annotations=None, index=None, reasoning_index=None):
            """Render the sample at position pos. Returns tuple of component values."""
            if not filtered or pos < 0 or pos >= len(filtered):
                return ("*No samples match filters*",
                        None, [], "*No rep info*", gr.update(choices=[], value=None),
                        "", "", "*No exercise info*", "*No metrics*",
                        "*No video selected*", {},
                        None, "", "", "*No reasoning*",
                        "*No judge verdict for any rep referenced by this sample*")

            sample = samples[filtered[pos]]
            parsed = parse_sample(sample)

            question_md = render_question(parsed)
            verification_md = render_verification(parsed)
            distractor_md = render_distractors(parsed)

            # Determine display mode based on question template
            video_id = parsed["video_id"]
            template = parsed["template"]
            is_rom_comparison = (
                template in {"tier_b_rom_comparison", "tier_b_peak_comparison"}
                and parsed["rep_comparison"]
            )
            _ALLREPS_TEMPLATES = {
                "tier_b_error_detection", "tier_b_correctness_criteria",
                "tier_b_compensatory",
                "tier_c_kinematic_trend", "tier_c_variability_analysis",
                "tier_c_trend_analysis", "tier_c_error_trend",
                "tier_d_side_comparison", "tier_d_asymmetry_detection",
                "tier_d_side_consistency", "tier_d_peak_comparison",
                # 3D multi-rep templates — surface all relevant reps in the
                # player so reviewers can compare what each rep contributed.
                "tier_b_axial_vs_lean_comparison",
                "tier_b_compensatory_3d",
                "tier_c_coordination_3d",
            }
            auto_allreps = template in _ALLREPS_TEMPLATES

            # Per-video question numbering
            abs_idx = filtered[pos]
            q_numbering = (index or {}).get("q_numbering", {})
            q_num, q_total = q_numbering.get(abs_idx, (0, 0))
            q_label = f" | **Q**: {q_num}/{q_total}" if q_num else ""

            # Use production video_frames when available — these are the exact cropped
            # frames the model saw, already sliced to the relevant rep(s) for this question.
            prod_frames = parsed.get("video_frames") or []
            prod_fps = parsed.get("fps") or _VIDEO_FPS_CACHE.get(video_id) or CONFIG["default_fps"]
            need_flip = _VIDEO_FLIP_CACHE.get(video_id, False)

            if prod_frames and not skeleton and override_rep is None and not all_reps:
                # Fast path: use the canonical frame list directly from the JSONL row.
                cache_key = f"prod_{video_id}_{abs_idx}"
                video_path = get_or_create_video_from_frames(prod_frames, prod_fps, cache_key, hflip=need_flip)
                gallery = get_gallery_from_frames(prod_frames)
                n_frames = len(prod_frames)
                rep_idx = parsed["verification"].get("rep_index") or 1
                # For rom_comparison, surface the filesystem rep indices clearly
                _rc = parsed.get("rep_comparison") or {}
                if template in {"tier_b_rom_comparison", "tier_b_peak_comparison"} and _rc:
                    rep_info = (
                        f"**Video**: {video_id} | "
                        f"Rep 1 = fs rep {_rc.get('rep_a','?')} | Rep 2 = fs rep {_rc.get('rep_b','?')} | "
                        f"{n_frames} frames (production){q_label}"
                    )
                else:
                    rep_info = f"**Video**: {video_id} | {n_frames} frames (production){q_label}"
            elif is_rom_comparison and not all_reps:
                # Show exactly 2 reps labeled "Rep 1" and "Rep 2"
                rep_a = parsed["rep_comparison"]["rep_a"]
                rep_b = parsed["rep_comparison"]["rep_b"]
                video_path = get_or_create_2rep_video(
                    video_id, rep_a, rep_b, skeleton=skeleton)
                gallery = get_2rep_gallery_frames(
                    video_id, rep_a, rep_b, skeleton=skeleton)
                rep_idx = rep_a
                rep_info = (
                    f"**Video**: {video_id} | "
                    f"**Reps**: {rep_a} & {rep_b} (shown as Rep 1 & Rep 2){q_label}"
                )
            elif all_reps or auto_allreps:
                rep_idx = 1  # fallback used by get_video_metadata below
                verification = parsed.get("verification") or parsed.get("metadata", {}).get("verification", {})
                q_rep_indices = verification.get("rep_indices") or []
                if auto_allreps and not all_reps and q_rep_indices:
                    # Show only the specific reps used for this question
                    rep_idx = q_rep_indices[0]
                    sel_paths, sel_boundaries = get_selected_reps_frame_paths(
                        video_id, q_rep_indices, use_cropped=not skeleton, skeleton=skeleton)
                    if sel_paths:
                        fps_val = get_video_fps(video_id, str(resolve_video_dir(video_id) / "repetitions" / f"repetition_{q_rep_indices[0]}"), prefer_dir=True)
                        cache_key = f"sel_{video_id}_{'_'.join(str(r) for r in sorted(q_rep_indices))}{'_skel' if skeleton else ''}_fpsfix"
                        video_path = get_or_create_video_from_frames(sel_paths, fps_val, cache_key, hflip=need_flip)
                        gallery = get_gallery_from_frames(sel_paths)
                        n_analyzed = len(sel_boundaries)
                        rep_info = f"**Video**: {video_id} | **Reps**: {', '.join(str(r) for r in sorted(q_rep_indices))} ({n_analyzed} analyzed){q_label}"
                    else:
                        # Fallback to full video if specific reps not found
                        video_path = get_or_create_video(
                            video_id, 1, skeleton=skeleton, all_reps=True, use_cropped=not skeleton)
                        gallery = get_gallery_frames(
                            video_id, 1, skeleton=skeleton, all_reps=True, use_cropped=not skeleton)
                        _, boundaries = get_allreps_frame_paths(video_id, use_cropped=not skeleton)
                        rep_info = f"**Video**: {video_id} | **All {len(boundaries)} reps**{q_label}"
                else:
                    # User explicitly selected "All Reps" — show full video
                    available = _get_available_reps(video_id)
                    rep_idx = int(available[0]) if available else 1
                    video_path = get_or_create_video(
                        video_id, rep_idx, skeleton=skeleton, all_reps=True, use_cropped=not skeleton)
                    gallery = get_gallery_frames(
                        video_id, rep_idx, skeleton=skeleton, all_reps=True, use_cropped=not skeleton)
                    _, boundaries = get_allreps_frame_paths(video_id, use_cropped=not skeleton)
                    total_reps = len(boundaries)
                    analyzed = verification.get("num_reps_analyzed") or (len(verification.get("rep_indices") or []) or None)
                    reps_label = f"All {total_reps} reps" if analyzed is None else f"All {total_reps} reps ({analyzed} analyzed)"
                    rep_info = f"**Video**: {video_id} | **{reps_label}**{q_label}"
            else:
                # Single rep mode (Tier A default or manual rep override) — always use cropped
                rep_idx = 1
                if override_rep is not None:
                    rep_idx = override_rep
                elif parsed["verification"].get("rep_index"):
                    rep_idx = parsed["verification"]["rep_index"]
                # Use _frame_list when present — explicit frame sequence stored in
                # the JSONL (paths into cropped_images/). Includes tail frames from
                # next rep for Tier A to complete the return phase.
                # When skeleton=True, remap each frame to the cropped_images_keypoints/
                # equivalent by filename so the same frames are shown with overlay.
                frame_list = samples[filtered[pos]].get("_frame_list") or []
                if frame_list and override_rep is None:
                    if skeleton:
                        skel_src = _frame_source_dir(video_id, skeleton=True)
                        remapped = []
                        for f in frame_list:
                            name = Path(f).name
                            p = skel_src / name
                            if p.is_file():
                                remapped.append(str(p))
                        valid_frames = remapped
                    else:
                        valid_frames = [f for f in frame_list if Path(f).is_file()]
                    fps_val = get_video_fps(video_id, str(resolve_video_dir(video_id) / "repetitions" / f"repetition_{rep_idx}"), prefer_dir=True)
                    cache_key = f"flist_{video_id}_rep{rep_idx}{'_skel' if skeleton else ''}{'_flip' if need_flip else ''}_fpsfix"
                    video_path = get_or_create_video_from_frames(valid_frames, fps_val, cache_key, hflip=need_flip)
                    gallery = get_gallery_from_frames(valid_frames)
                else:
                    video_path = get_or_create_video(
                        video_id, rep_idx, skeleton=skeleton, all_reps=False, use_cropped=not skeleton)
                    gallery = get_gallery_frames(
                        video_id, rep_idx, skeleton=skeleton, all_reps=False, use_cropped=not skeleton)
                rep_info = f"**Video**: {video_id}{q_label}"

            # Show question-relevant reps in the dropdown
            v_tmp = parsed["verification"]
            rc_tmp = parsed.get("rep_comparison") or {}
            if template in {"tier_b_rom_comparison", "tier_b_peak_comparison"} and rc_tmp:
                question_reps = [str(rc_tmp["rep_a"]), str(rc_tmp["rep_b"])]
            elif v_tmp.get("rep_index") is not None:
                question_reps = [str(v_tmp["rep_index"])]
            elif v_tmp.get("rep_indices"):
                question_reps = [str(r) for r in sorted(v_tmp["rep_indices"])]
            else:
                question_reps = _get_available_reps(video_id)
            rep_dropdown = gr.update(
                choices=question_reps,
                value=question_reps[0] if question_reps else None,
            )

            exercise_md = get_exercise_detail_for_video(video_id)
            vm = get_video_metrics_cached(video_id)
            video_phases = get_video_phases_cached(video_id) if vm else None
            # Determine which filesystem rep index/indices this question is about
            v = parsed["verification"]
            rc = parsed.get("rep_comparison") or {}
            if template in {"tier_b_rom_comparison", "tier_b_peak_comparison"} and rc:
                highlight_rep = None
                highlight_reps = {
                    rc["rep_a"]: f"Rep 1 (fs:{rc['rep_a']})",
                    rc["rep_b"]: f"Rep 2 (fs:{rc['rep_b']})",
                }
            elif v.get("rep_indices") and len(v["rep_indices"]) > 1:
                # Tier C/D/compensatory: highlight all analyzed reps with a star
                highlight_rep = None
                highlight_reps = {ri: f"★" for ri in v["rep_indices"]}
            else:
                highlight_rep = v.get("rep_index")
                highlight_reps = None

            # For Tier D, pass side-filtered means so the table footer matches
            # the question values (which use _split_by_side, not raw session mean).
            tier = parsed["metadata"].get("difficulty_tier", "")
            tier_d_side_means = None
            primary_joint = parsed.get("joint", "")
            priority_joints = [primary_joint] if primary_joint else []
            if "TIER_D" in tier and vm:
                right_joint = primary_joint.replace("left_", "right_", 1) if primary_joint.startswith("left_") else primary_joint
                if right_joint != primary_joint:
                    priority_joints = [primary_joint, right_joint]
                lm = v.get("left_mean_rom_deg") or v.get("left_mean_peak_deg")
                rm = v.get("right_mean_rom_deg") or v.get("right_mean_peak_deg")
                if lm is not None and rm is not None:
                    tier_d_side_means = {primary_joint: (lm, rm), right_joint: (lm, rm)}
            elif template in {"tier_b_rom_comparison", "tier_b_peak_comparison"} and rc:
                priority_joints = [joint for joint in [rc.get("joint"), primary_joint] if joint]

            _active_side_per_rep = parsed["metadata"].get("active_side_per_rep")

            # Compute the rep window the question uses, so the metrics table
            # can hide rows that the question does NOT reference. This matches
            # what the question/answer text says against what the table shows.
            _restrict_reps: Optional[List[int]] = None
            try:
                v_local = parsed["verification"] or {}
                rc_local = parsed.get("rep_comparison") or {}
                # Multi-rep templates: rep_indices > selected_repetition_ids > rep_a/rep_b > rep_index
                if v_local.get("rep_indices"):
                    _restrict_reps = [int(x) for x in v_local["rep_indices"]]
                else:
                    _prod_local = parsed["metadata"].get("_production", {}) or {}
                    sel = _prod_local.get("selected_repetition_ids") or []
                    if sel:
                        _restrict_reps = [int(s.replace("repetition_", "")) for s in sel if str(s).startswith("repetition_")]
                    elif rc_local.get("rep_a") is not None and rc_local.get("rep_b") is not None:
                        _restrict_reps = [int(rc_local["rep_a"]), int(rc_local["rep_b"])]
                    elif v_local.get("rep_index"):
                        _restrict_reps = [int(v_local["rep_index"])]
            except (KeyError, ValueError, TypeError):
                _restrict_reps = None

            # 3D-template questions: show ONLY the 3D feature table, not the
            # 2D kinematic table — the latter's per-joint angles don't drive
            # the answer for pose_class / motion_plane / trunk_axial_yaw /
            # trunk_lean / neck_rotation / active_side_3d / coordination_3d.
            _3D_TEMPLATES = {
                "tier_a_pose_class",
                "tier_a_motion_plane",
                "tier_a_active_side_3d",
                "tier_a_neck_rotation",
                "tier_a_neck_flexion_3d",
        "tier_a_neck_flexion_3d",
                "tier_a_trunk_axial_yaw_rom",
                "tier_a_trunk_rotation_direction",
                "tier_a_trunk_lean_direction",
                "tier_a_trunk_sagittal_direction",
                "tier_a_trunk_sagittal_rom",
                "tier_a_hip_flexion_3d",
                "tier_a_hip_abduction_rom_3d",
                "tier_a_hip_hinge_rom_3d",
                "tier_a_limb_extension_arm_3d",
        "tier_a_limb_extension_leg_3d",
        "tier_a_limb_extension_arm_3d",
        "tier_a_limb_extension_leg_3d",
        "tier_a_hip_hinge_rom_3d",
        "tier_a_limb_extension_arm_3d",
        "tier_a_limb_extension_leg_3d",
        "tier_a_hip_abduction_rom_3d",
        "tier_a_hip_hinge_rom_3d",
        "tier_a_limb_extension_arm_3d",
        "tier_a_limb_extension_leg_3d",
                "tier_a_knee_flexion_3d",
                "tier_a_knee_pushup_3d",
                "tier_a_quad_stretch_3d",
        "tier_a_trunk_side_bend_3d",
        "tier_a_sidelying_abduction_3d",
        "tier_a_standing_row_3d",
                "tier_a_shoulder_extension_3d",
                "tier_a_lower_body_depth_3d",
                "tier_a_shoulder_er_3d",
                "tier_a_shoulder_elevation_rom_3d",
                "tier_a_elbow_flexion_rom_3d",
                "tier_a_wrist_rom_3d",
                "tier_a_trunk_stability_hold_3d",
                "tier_a_plank_back_bend_3d",
                "tier_b_axial_vs_lean_comparison",
                "tier_b_compensatory_3d",
                "tier_c_coordination_3d",
            }
            _is_3d = (parsed.get("template") or "") in _3D_TEMPLATES
            if _is_3d:
                _vid = parsed.get("video_id", "")
                pose3d_md = render_pose3d_metrics(_vid, restrict_to_reps=_restrict_reps)
                metrics_md = pose3d_md or "*No 3D feature data available for this video.*"
            else:
                metrics_md = render_metrics_table(vm, highlight_joint=primary_joint, phases=video_phases, highlight_rep=highlight_rep, highlight_reps=highlight_reps, tier_d_side_means=tier_d_side_means, priority_joints=priority_joints, active_side_per_rep=_active_side_per_rep, restrict_to_reps=_restrict_reps) if vm else "*No metrics available*"

            # Determine question-relevant reps for metadata display.
            # _production.selected_repetition_ids is the authoritative source for
            # multi-rep questions (tier_b/c templates that span several reps).
            _prod = parsed["metadata"].get("_production", {}) or {}
            _sel_rep_ids = _prod.get("selected_repetition_ids")
            _MULTIREP_TEMPLATES = {
                "tier_b_error_detection", "tier_b_correctness_criteria",
                "tier_b_compensatory",
                "tier_c_kinematic_trend", "tier_c_variability_analysis",
                "tier_c_trend_analysis", "tier_c_error_trend",
                "tier_d_side_comparison", "tier_d_asymmetry_detection",
                "tier_d_side_consistency", "tier_d_peak_comparison",
            }
            if template in _MULTIREP_TEMPLATES and _sel_rep_ids:
                q_reps_for_meta = _sel_rep_ids  # list of "repetition_N" strings
            elif template in {"tier_b_rom_comparison", "tier_b_peak_comparison"} and rc:
                q_reps_for_meta = [f"repetition_{rc['rep_a']}", f"repetition_{rc['rep_b']}"]
            elif v.get("rep_index"):
                q_reps_for_meta = [f"repetition_{v['rep_index']}"]
            else:
                q_reps_for_meta = None
            video_meta = get_video_metadata(video_id, rep_idx, question_reps=q_reps_for_meta)

            # Look up existing annotation for this sample
            key = _sample_key(parsed)
            ann = (annotations or {}).get(key)
            ann_rating = ann["rating"] if ann else None
            ann_note = ann.get("note", "") if ann else ""
            if ann:
                _rating = ann["rating"]
                _emoji = {"Good quality": "✓", "Maybe redundant": "~", "Has issues": "⚠"}.get(_rating, "")
                _body = f"**{_emoji} Previously annotated: {_rating}**"
                if ann_note:
                    _body += f"\n\n> {ann_note}"
                ann_status = _body
            else:
                ann_status = ""

            reasoning_entry = lookup_reasoning(sample, reasoning_index or {})
            reasoning_md = render_reasoning(reasoning_entry, sample)

            # Full QA Metadata: everything coupled to the question in the JSONL record.
            # Includes metadata, frame list, and live per-rep metrics from VideoMetrics.
            full_meta = {
                "metadata": parsed["metadata"],
                "_frame_list": parsed.get("_frame_list", []),
            }
            if vm:
                def _rep_active_side(joint_name, rep_index, asp):
                    if asp is None:
                        return None
                    if "all" in asp:
                        return asp["all"]
                    return asp.get(str(rep_index))

                full_meta["_per_rep_metrics"] = {
                    joint: [
                        {
                            "rep_index": rm.rep_index,
                            "active_side": _rep_active_side(joint, rm.rep_index, _active_side_per_rep),
                            "rom_degrees": round(rm.rom_degrees, 2),
                            "peak_angle_degrees": round(rm.peak_angle_degrees, 2),
                            "min_angle_degrees": round(rm.min_angle_degrees, 2),
                            "start_angle_degrees": round(rm.start_angle_degrees, 2),
                            "end_angle_degrees": round(rm.end_angle_degrees, 2),
                            "mean_angle_degrees": round(rm.mean_angle_degrees, 2),
                            "num_frames": rm.num_frames,
                            "peak_frame": rm.peak_frame,
                            "min_frame": rm.min_frame,
                            "fps": round(getattr(rm, "fps", 0.0), 2),
                            "concentric_velocity": round(rm.concentric_velocity, 2),
                            "eccentric_velocity": round(rm.eccentric_velocity, 2),
                            "velocity_ratio": round(rm.velocity_ratio, 2),
                            "hold_frames": rm.hold_frames,
                            "hold_time_seconds": round(getattr(rm, "hold_time_seconds", 0.0), 2),
                        }
                        for rm in rep_list
                    ]
                    for joint, rep_list in vm.per_rep.items()
                    if rep_list
                }

            judge_md = render_judge_panel(sample)

            return (question_md, video_path, gallery, rep_info, rep_dropdown,
                    verification_md, distractor_md, exercise_md, metrics_md,
                    video_meta, full_meta,
                    ann_rating, ann_note, ann_status, reasoning_md,
                    judge_md)

        load_btn.click(
            fn=on_load,
            inputs=[jsonl_input, mcqa_skeleton_cb, mcqa_allreps_cb],
            outputs=[
                samples_state, index_state, filtered_state, pos_state,
                annotations_state, annotations_path_state, reasoning_index_state,
                sample_counter,
                tier_filter, joint_filter, exercise_filter, body_region_filter, kp_source_filter, source_dataset_filter, exercise_id_filter, exercise_type_filter, category_filter,
                annotated_by_filter,
                mcqa_question, mcqa_video, mcqa_gallery, mcqa_rep_info, mcqa_rep_selector,
                mcqa_verification, mcqa_distractors, mcqa_exercise_detail, mcqa_metrics_md,
                mcqa_video_meta, mcqa_metadata_json,
                audit_rating, audit_note, audit_status, mcqa_reasoning_md,
                mcqa_judge_md,
                tier_chart, joint_chart, distractor_chart,
                ex_type_template_chart, ex_type_tier_chart,
                audit_counter,
            ],
        )

        # ---- Dataset picker: selecting a name fills the path and triggers load ----
        _load_outputs = [
            samples_state, index_state, filtered_state, pos_state,
            annotations_state, annotations_path_state, reasoning_index_state,
            sample_counter,
            tier_filter, joint_filter, exercise_filter, body_region_filter, kp_source_filter, source_dataset_filter, exercise_id_filter, exercise_type_filter, category_filter,
            annotated_by_filter,
            mcqa_question, mcqa_video, mcqa_gallery, mcqa_rep_info, mcqa_rep_selector,
            mcqa_verification, mcqa_distractors, mcqa_exercise_detail, mcqa_metrics_md,
            mcqa_video_meta, mcqa_metadata_json,
            audit_rating, audit_note, audit_status, mcqa_reasoning_md,
            mcqa_judge_md,
            tier_chart, joint_chart, distractor_chart,
            ex_type_template_chart, ex_type_tier_chart,
            audit_counter,
        ]

        def on_dataset_pick(label, skeleton, all_reps):
            if not label:
                return [gr.update()] * len(_load_outputs)
            path = _browse_paths.get(label, "")
            if not path:
                return [gr.update()] * len(_load_outputs)
            return on_load(path, skeleton, all_reps)

        dataset_picker.change(
            fn=on_dataset_pick,
            inputs=[dataset_picker, mcqa_skeleton_cb, mcqa_allreps_cb],
            outputs=_load_outputs,
        )

        # ---- MCQA: Filter ----
        def on_filter(samples, index, tier, joint, exercise, movement_label, body_region, kp_source, source_dataset, exercise_id, exercise_type, category, frames_source, camera_perspective, annotation, annotated_by, diff_status, judge_pass1, judge_pass2, salvage_origin, geometry_assessment, skeleton, all_reps, annotations):
            if not samples or not isinstance(samples, list):
                return [], 0, "*No samples*", *render_sample([], [], 0)

            filtered = get_filtered_indices(
                index, len(samples), tier, joint, exercise,
                movement_label=movement_label, samples=samples,
                body_region=body_region,
                kp_source=kp_source,
                source_dataset=source_dataset,
                exercise_id=exercise_id, exercise_type=exercise_type, category=category,
                frames_source=frames_source,
                camera_perspective=camera_perspective,
                annotation=annotation, annotations=annotations,
                annotated_by=annotated_by,
                diff_status=diff_status,
                judge_pass1=judge_pass1,
                judge_pass2=judge_pass2,
                salvage_origin=salvage_origin,
                geometry_assessment=geometry_assessment,
            )
            if not filtered:
                return [], 0, "*No samples match filters*", *render_sample(samples, [], 0)

            # Use module-level _REASONING_INDEX (avoids Gradio cascade input-count bug)
            result = render_sample(samples, filtered, 0, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=index)
            return filtered, 0, f"**Sample 1 / {len(filtered)}**", *result

        filter_outputs = [
            filtered_state, pos_state, sample_counter,
            mcqa_question, mcqa_video, mcqa_gallery, mcqa_rep_info, mcqa_rep_selector,
            mcqa_verification, mcqa_distractors, mcqa_exercise_detail, mcqa_metrics_md,
            mcqa_video_meta, mcqa_metadata_json,
            audit_rating, audit_note, audit_status, mcqa_reasoning_md,
            mcqa_judge_md,
        ]

        for filt in [tier_filter, joint_filter, exercise_filter, label_filter, body_region_filter, kp_source_filter, source_dataset_filter, exercise_id_filter, exercise_type_filter, category_filter, frames_source_filter, camera_perspective_filter, annotation_filter, annotated_by_filter, diff_status_filter, judge_pass1_filter, judge_pass2_filter, salvage_origin_filter, geometry_assessment_filter]:
            filt.change(
                fn=on_filter,
                inputs=[samples_state, index_state, tier_filter, joint_filter, exercise_filter, label_filter, body_region_filter, kp_source_filter, source_dataset_filter, exercise_id_filter, exercise_type_filter, category_filter, frames_source_filter, camera_perspective_filter, annotation_filter, annotated_by_filter, diff_status_filter, judge_pass1_filter, judge_pass2_filter, salvage_origin_filter, geometry_assessment_filter, mcqa_skeleton_cb, mcqa_allreps_cb, annotations_state],
                outputs=filter_outputs,
            )

        # ---- MCQA: Navigation ----
        def navigate(samples, filtered, pos, skeleton, all_reps, annotations, idx, reas_idx, delta):
            if not filtered:
                return pos, "*No samples*", *render_sample([], [], 0)
            new_pos = max(0, min(len(filtered) - 1, pos + delta))
            result = render_sample(samples, filtered, new_pos, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=idx, reasoning_index=reas_idx)
            return new_pos, f"**Sample {new_pos + 1} / {len(filtered)}**", *result

        nav_outputs = [
            pos_state, sample_counter,
            mcqa_question, mcqa_video, mcqa_gallery, mcqa_rep_info, mcqa_rep_selector,
            mcqa_verification, mcqa_distractors, mcqa_exercise_detail, mcqa_metrics_md,
            mcqa_video_meta, mcqa_metadata_json,
            audit_rating, audit_note, audit_status, mcqa_reasoning_md,
            mcqa_judge_md,
        ]

        _nav_inputs = [samples_state, filtered_state, pos_state, mcqa_skeleton_cb, mcqa_allreps_cb, annotations_state, index_state, reasoning_index_state]

        prev_btn.click(
            fn=lambda s, f, p, sk, ar, ann, idx, ri: navigate(s, f, p, sk, ar, ann, idx, ri, -1),
            inputs=_nav_inputs,
            outputs=nav_outputs,
        )
        next_btn.click(
            fn=lambda s, f, p, sk, ar, ann, idx, ri: navigate(s, f, p, sk, ar, ann, idx, ri, 1),
            inputs=_nav_inputs,
            outputs=nav_outputs,
        )
        random_btn.click(
            fn=lambda s, f, p, sk, ar, ann, idx, ri: navigate(s, f, 0, sk, ar, ann, idx, ri, random.randint(0, max(0, len(f) - 1))) if f else (p, "*No samples*", *render_sample([], [], 0)),
            inputs=_nav_inputs,
            outputs=nav_outputs,
        )

        def on_refresh(samples, filtered, pos, skeleton, all_reps, annotations, idx, reas_idx):
            if not filtered:
                return pos, "*No samples*", *render_sample([], [], 0)
            result = render_sample(samples, filtered, pos, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=idx, reasoning_index=reas_idx)
            return pos, f"**Sample {pos + 1} / {len(filtered)}**", *result

        refresh_btn.click(
            fn=on_refresh,
            inputs=_nav_inputs,
            outputs=nav_outputs,
        )

        # ---- MCQA: Clear filters — deselect every dropdown back to defaults.
        # Tier filter is a multi-select (default = empty list); the rest are
        # single-select with default value "All".
        def _clear_filters():
            return (
                [],      # tier (multi-select)
                "All",   # joint
                "All",   # exercise
                "All",   # label
                "All",   # body_region
                "All",   # kp_source
                "All",   # source_dataset
                "All",   # exercise_id
                "All",   # exercise_type
                "All",   # category
                "All",   # frames_source
                "All",   # camera_perspective
                "All",   # annotation
                "All",   # annotated_by
                "All",   # diff_status
                "All",   # judge_pass1
                "All",   # judge_pass2
                "All",   # salvage_origin
                "All",   # geometry_assessment
            )

        clear_filters_btn.click(
            fn=_clear_filters,
            inputs=[],
            outputs=[
                tier_filter, joint_filter, exercise_filter, label_filter,
                body_region_filter, kp_source_filter, source_dataset_filter,
                exercise_id_filter,
                exercise_type_filter, category_filter, frames_source_filter,
                camera_perspective_filter,
                annotation_filter, annotated_by_filter, diff_status_filter,
                judge_pass1_filter, judge_pass2_filter, salvage_origin_filter,
                geometry_assessment_filter,
            ],
        )

        # ---- MCQA: Display toggle re-renders current sample ----
        def on_mcqa_display_toggle(samples, filtered, pos, skeleton, all_reps, annotations, idx, reas_idx):
            if not filtered:
                return render_sample([], [], 0)
            return render_sample(samples, filtered, pos, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=idx, reasoning_index=reas_idx)

        _toggle_inputs = [samples_state, filtered_state, pos_state, mcqa_skeleton_cb, mcqa_allreps_cb, annotations_state, index_state, reasoning_index_state]

        _toggle_outputs = [
            mcqa_question, mcqa_video, mcqa_gallery, mcqa_rep_info, mcqa_rep_selector,
            mcqa_verification, mcqa_distractors, mcqa_exercise_detail, mcqa_metrics_md,
            mcqa_video_meta, mcqa_metadata_json,
            audit_rating, audit_note, audit_status, mcqa_reasoning_md,
            mcqa_judge_md,
        ]

        mcqa_skeleton_cb.change(
            fn=on_mcqa_display_toggle,
            inputs=_toggle_inputs,
            outputs=_toggle_outputs,
        )
        mcqa_allreps_cb.change(
            fn=on_mcqa_display_toggle,
            inputs=_toggle_inputs,
            outputs=_toggle_outputs,
        )

        # ---- MCQA: side-by-side comparison panel ----
        def on_v6_load(path, samples):
            """Load the comparison JSONL and rebuild the index so v6_audit
            tags propagate from comparison-side samples onto primary-side ones
            via the slot-key match."""
            n = load_v6_compare_index((path or "").strip())
            if n == 0:
                return "*No comparison loaded* (path empty or file missing)", gr.update()
            status = f"Loaded **{n}** comparison samples."
            if isinstance(samples, list) and samples:
                new_index = build_index(samples)
                status += " v6 Audit filter index rebuilt."
                return status, new_index
            return status, gr.update()

        def on_v6_compare_refresh(samples, filtered, pos):
            """Render side-by-side comparison for the current sample."""
            if not _V6_COMPARE_INDEX:
                return "*Load a comparison JSONL to enable side-by-side comparison*"
            if not samples or not filtered or pos < 0 or pos >= len(filtered):
                return "*No sample selected*"
            try:
                return render_v6_comparison(samples[filtered[pos]])
            except Exception as e:
                return f"*comparison error: {e}*"

        v6_load_btn.click(
            fn=on_v6_load,
            inputs=[v6_jsonl_input, samples_state],
            outputs=[v6_load_status, index_state],
        ).then(
            fn=on_v6_compare_refresh,
            inputs=[samples_state, filtered_state, pos_state],
            outputs=[v6_compare_md],
        )

        # Refresh comparison whenever the position changes.
        pos_state.change(
            fn=on_v6_compare_refresh,
            inputs=[samples_state, filtered_state, pos_state],
            outputs=[v6_compare_md],
        )
        filtered_state.change(
            fn=on_v6_compare_refresh,
            inputs=[samples_state, filtered_state, pos_state],
            outputs=[v6_compare_md],
        )

        # ---- MCQA: Video search ----
        def on_video_search(query, samples, filtered, skeleton, all_reps, annotations, idx, reas_idx):
            """Jump to the first sample matching a video_id substring."""
            query = query.strip()
            if not query or not samples:
                return gr.update(), gr.update(), *render_sample(samples, filtered, 0)
            for i, sample_idx in enumerate(filtered):
                vid = samples[sample_idx].get("metadata", {}).get("video_id", "")
                if query in vid:
                    result = render_sample(samples, filtered, i, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=idx, reasoning_index=reas_idx)
                    return i, f"**Sample {i + 1} / {len(filtered)}**", *result
            gr.Warning(f"No sample found matching '{query}'")
            return gr.update(), gr.update(), *render_sample(samples, filtered, 0)

        video_search_btn.click(
            fn=on_video_search,
            inputs=[video_search, samples_state, filtered_state, mcqa_skeleton_cb, mcqa_allreps_cb, annotations_state, index_state, reasoning_index_state],
            outputs=nav_outputs,
        )
        video_search.submit(
            fn=on_video_search,
            inputs=[video_search, samples_state, filtered_state, mcqa_skeleton_cb, mcqa_allreps_cb, annotations_state, index_state, reasoning_index_state],
            outputs=nav_outputs,
        )

        # ---- MCQA: Rep selector ----
        def on_rep_change(rep_value, samples, filtered, pos, skeleton, all_reps, annotations, idx, reas_idx):
            """Re-render the current sample with a different rep."""
            if rep_value is None or rep_value == "" or not filtered:
                return render_sample(samples, filtered, pos, annotations=annotations, index=idx, reasoning_index=reas_idx)
            sample = samples[filtered[pos]]
            tmpl = sample.get("metadata", {}).get("question_template", "")
            if tmpl in {"tier_b_rom_comparison", "tier_b_peak_comparison"} and sample.get("metadata", {}).get("rep_comparison"):
                return render_sample(samples, filtered, pos, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=idx, reasoning_index=reas_idx)
            if tmpl in {"tier_b_error_detection", "tier_b_correctness_criteria", "tier_b_compensatory"}:
                return render_sample(samples, filtered, pos, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=idx, reasoning_index=reas_idx)
            try:
                rep_idx = int(rep_value)
            except (ValueError, TypeError):
                return render_sample(samples, filtered, pos, skeleton=skeleton, all_reps=all_reps, annotations=annotations, index=idx, reasoning_index=reas_idx)
            return render_sample(samples, filtered, pos, skeleton=skeleton, all_reps=all_reps, override_rep=rep_idx, annotations=annotations, index=idx, reasoning_index=reas_idx)

        mcqa_rep_selector.change(
            fn=on_rep_change,
            inputs=[mcqa_rep_selector, samples_state, filtered_state, pos_state, mcqa_skeleton_cb, mcqa_allreps_cb, annotations_state, index_state, reasoning_index_state],
            outputs=_toggle_outputs,
        )

        # ---- MCQA: Quality Audit ----
        def on_save_annotation(rating, note, samples, filtered, pos, annotations, jsonl_path, idx):
            if not filtered or not rating:
                return annotations, "*Select a rating first*", gr.update()
            abs_idx = filtered[pos]
            sample = samples[abs_idx]
            parsed = parse_sample(sample)
            key = _sample_key(parsed)
            q_num, q_total = (idx or {}).get("q_numbering", {}).get(abs_idx, (0, 0))
            entry = {
                "rating": rating,
                "note": note or "",
                "video_id": parsed["video_id"],
                "video_path": parsed["video_path"],
                "dataset_version": _dataset_version(jsonl_path),
                "exercise_name": parsed["exercise_name"],
                "tier": parsed["tier"],
                "template": parsed["template"],
                "joint": parsed["joint"],
                "question_index": q_num,
                "questions_in_video": q_total,
                "timestamp": datetime.now().isoformat(),
                # Tag the author so multiple reviewers using the same shared
                # annotations file can be distinguished. Override via env var:
                #   ANNOTATION_AUTHOR=alice python app.py ...
                "author": os.environ.get("ANNOTATION_AUTHOR", os.environ.get("USER", "unknown")),
            }
            updated = _save_annotation(jsonl_path, key, entry, annotations)
            return updated, f"Saved! (**{rating}**)", _annotation_counter_md(updated)

        audit_save_btn.click(
            fn=on_save_annotation,
            inputs=[audit_rating, audit_note, samples_state, filtered_state, pos_state, annotations_state, annotations_path_state, index_state],
            outputs=[annotations_state, audit_status, audit_counter],
        )

        # ---- Exclude question (separate from annotations) ----
        # Appends a JSON line to the repo's training/excluded_questions.jsonl.
        # The build pipeline (apply_excluded_questions.py, stage 1.5) consumes
        # this file and removes matching samples on the next rebuild.
        # Falls back to the JSONL's directory if the repo path isn't writable.
        def on_exclude_question(samples, filtered, pos, jsonl_path, note):
            if not filtered:
                return "*No question selected*"
            sample = samples[filtered[pos]]
            md = sample.get("metadata", {}) or {}
            ts = md.get("generation_timestamp", "")
            if not ts:
                return "*Sample has no generation_timestamp — cannot exclude reliably*"
            repo_excl = Path(
                "/home/sgsilva/vlm-post-training/aux_tasks/video_tasks/"
                "video_mcqa/training/excluded_questions.jsonl"
            )
            if repo_excl.parent.is_dir():
                out_path = repo_excl
            else:
                jsonl_p = Path(jsonl_path) if jsonl_path else None
                out_dir = jsonl_p.parent if jsonl_p and jsonl_p.parent.exists() else Path.cwd()
                out_path = out_dir / "excluded_questions.jsonl"
            entry = {
                "generation_timestamp": ts,
                "video_id": md.get("video_id", ""),
                "exercise_code": md.get("exercise_code", ""),
                "question_template": md.get("question_template", ""),
                "joint": (md.get("verification") or {}).get("joint", ""),
                "correct_text": md.get("correct_text", ""),
                "dataset_version": _dataset_version(jsonl_path or ""),
                "note": note or "",
                "excluded_at": datetime.now().isoformat(),
            }
            try:
                with out_path.open("a") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                return f"*Failed to write {out_path}: {e}*"
            return f"✓ Excluded · written to `{out_path}`"

        exclude_btn.click(
            fn=on_exclude_question,
            inputs=[samples_state, filtered_state, pos_state, annotations_path_state, audit_note],
            outputs=[exclude_status],
        )

        # ---- Exercise Explorer: filter + search ----
        def filter_exercises(region, search_text):
            df = EXERCISE_DF
            if df.empty:
                return []

            if region != "All":
                df = df[df["body_region"] == region]

            if search_text.strip():
                mask = df["exercise_name"].str.contains(search_text.strip(), case=False, na=False)
                df = df[mask]

            rows = []
            for _, row in df.iterrows():
                code = row.get("exercise_code", "")
                name = row.get("exercise_name", "")
                region_val = row.get("body_region", "")
                n_videos = len(EXERCISE_VIDEO_INDEX.get(str(code), []))
                rows.append([str(code), name, region_val, n_videos])

            return rows

        ex_region_filter.change(
            fn=filter_exercises,
            inputs=[ex_region_filter, ex_search],
            outputs=[ex_table],
        )
        ex_search.change(
            fn=filter_exercises,
            inputs=[ex_region_filter, ex_search],
            outputs=[ex_table],
        )

        # ---- Exercise Explorer: build video rows helper ----
        def _build_video_rows(code: str, label_filter: str = "All") -> Tuple[str, list]:
            """Build video table rows for an exercise code, filtered by label."""
            video_ids = EXERCISE_VIDEO_INDEX.get(code, [])

            video_rows = []
            for vid in video_ids[:100]:
                info = PROCESSING_REPORT.get(vid, {})
                n_reps = info.get("num_repetitions", "?")
                label = get_video_label_summary(vid)
                if label_filter != "All" and label != label_filter:
                    continue
                video_rows.append([vid, n_reps, label, code])

            total = len(video_ids)
            shown = len(video_rows)
            if label_filter != "All":
                videos_label = f"### Videos ({shown} / {total}, filter: {label_filter})"
            else:
                videos_label = f"### Videos ({shown})"
            return videos_label, video_rows

        # ---- Exercise Explorer: select exercise ----
        def on_exercise_select(evt: gr.SelectData, table_data, label_filter):
            if not isinstance(table_data, pd.DataFrame) or table_data.empty:
                return "*No exercise selected*", "", [], "*No video*", ""

            row_idx = evt.index[0]
            if row_idx >= len(table_data):
                return "*Invalid selection*", "", [], "*No video*", ""

            code = str(table_data.iloc[row_idx, 0])

            if EXERCISE_DF.empty:
                return "*No exercise data*", "", [], "*No video*", ""

            matches = EXERCISE_DF[EXERCISE_DF["exercise_code"] == code]
            if matches.empty:
                return f"*Exercise {code} not found in CSV*", "", [], "*No video*", ""

            exercise_row = matches.iloc[0]
            detail_md = render_exercise_detail(exercise_row)
            videos_label, video_rows = _build_video_rows(code, label_filter)

            return detail_md, videos_label, video_rows, "*Select a video to view*", code

        ex_table.select(
            fn=on_exercise_select,
            inputs=[ex_table, ex_label_filter],
            outputs=[ex_detail, ex_videos_label, ex_video_table, ex_video_meta,
                     ex_exercise_code_state],
        )

        # ---- Exercise Explorer: label filter change ----
        def on_label_filter_change(label_filter, exercise_code):
            if not exercise_code:
                return "", []
            videos_label, video_rows = _build_video_rows(exercise_code, label_filter)
            return videos_label, video_rows

        ex_label_filter.change(
            fn=on_label_filter_change,
            inputs=[ex_label_filter, ex_exercise_code_state],
            outputs=[ex_videos_label, ex_video_table],
        )

        # ---- Exercise Explorer: select video ----
        def on_video_select(evt: gr.SelectData, video_table_data, skeleton, use_cropped):
            if not isinstance(video_table_data, pd.DataFrame) or video_table_data.empty:
                return None, [], "*No video*", gr.update(choices=[], value=None), ""

            row_idx = evt.index[0]
            if row_idx >= len(video_table_data):
                return None, [], "*Invalid selection*", gr.update(choices=[], value=None), ""

            video_id = str(video_table_data.iloc[row_idx, 0])

            # Discover available reps
            video_dir = resolve_video_dir(video_id)
            reps_dir = video_dir / "repetitions"
            rep_choices = []
            if reps_dir.exists():
                rep_choices = sorted(
                    [int(d.name.split("_")[1]) for d in reps_dir.iterdir()
                     if d.is_dir() and d.name.startswith("repetition_")],
                )
            if not rep_choices:
                rep_choices = [1]

            rep_idx = rep_choices[0]
            video_path = get_or_create_video(video_id, rep_index=rep_idx,
                                             skeleton=skeleton, use_cropped=use_cropped)
            gallery = get_gallery_frames(video_id, rep_index=rep_idx,
                                         skeleton=skeleton, use_cropped=use_cropped)
            meta_md = get_video_metadata(video_id, rep_index=rep_idx)

            # Show per-rep labels in dropdown
            vid_labels = get_video_labels(video_id)
            rep_display = []
            for r in rep_choices:
                lbl = vid_labels.get(r, "")
                tag = f" ({lbl})" if lbl else ""
                rep_display.append(f"{r}{tag}")
            return (video_path, gallery, meta_md,
                    gr.update(choices=rep_display, value=rep_display[0]),
                    video_id)

        ex_video_table.select(
            fn=on_video_select,
            inputs=[ex_video_table, ex_skeleton_cb, ex_cropped_cb],
            outputs=[ex_video_player, ex_gallery, ex_video_meta,
                     ex_rep_dropdown, ex_video_id_state],
        )

        # ---- Exercise Explorer: change rep / toggle skeleton / toggle cropped ----
        def on_display_change(rep_value, video_id, skeleton, use_cropped):
            if not video_id or not rep_value:
                return None, [], "*No video*"
            # Parse rep index from "1 (correct)" format
            rep_idx = int(rep_value.split(" ")[0].split("(")[0])
            video_path = get_or_create_video(video_id, rep_index=rep_idx,
                                             skeleton=skeleton, use_cropped=use_cropped)
            gallery = get_gallery_frames(video_id, rep_index=rep_idx,
                                         skeleton=skeleton, use_cropped=use_cropped)
            meta_md = get_video_metadata(video_id, rep_index=rep_idx)
            return video_path, gallery, meta_md

        _display_inputs = [ex_rep_dropdown, ex_video_id_state, ex_skeleton_cb, ex_cropped_cb]
        _display_outputs = [ex_video_player, ex_gallery, ex_video_meta]

        for trigger in [ex_rep_dropdown, ex_skeleton_cb, ex_cropped_cb]:
            trigger.change(
                fn=on_display_change,
                inputs=_display_inputs,
                outputs=_display_outputs,
            )

        # ---- Exercise Explorer: populate on startup ----
        def populate_exercises():
            return filter_exercises("All", "")

        app.load(fn=populate_exercises, outputs=[ex_table])

        # Auto-load default JSONL on startup
        def auto_load_jsonl():
            return on_load(CONFIG["default_jsonl"], False, False)

        app.load(
            fn=auto_load_jsonl,
            outputs=[
                samples_state, index_state, filtered_state, pos_state,
                annotations_state, annotations_path_state, reasoning_index_state,
                sample_counter,
                tier_filter, joint_filter, exercise_filter, body_region_filter, kp_source_filter, source_dataset_filter, exercise_id_filter, exercise_type_filter, category_filter,
                annotated_by_filter,
                mcqa_question, mcqa_video, mcqa_gallery, mcqa_rep_info, mcqa_rep_selector,
                mcqa_verification, mcqa_distractors, mcqa_exercise_detail, mcqa_metrics_md,
                mcqa_video_meta, mcqa_metadata_json,
                audit_rating, audit_note, audit_status, mcqa_reasoning_md,
                mcqa_judge_md,
                tier_chart, joint_chart, distractor_chart,
                ex_type_template_chart, ex_type_tier_chart,
                audit_counter,
            ],
        )

    return app


# ---------------------------------------------------------------------------
# 11. Main entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Video SFT Dataset Monitor")
    parser.add_argument("--port", type=int, default=CONFIG["port"])
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Video SFT Dataset Monitor")
    print("=" * 60)

    os.makedirs(CONFIG["video_cache_dir"], exist_ok=True)

    print("Loading global indexes...")
    init_global_indexes()

    print(f"\nStarting on port {args.port}...")
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=build_theme(),
        css=CUSTOM_CSS,
        allowed_paths=[
            CONFIG["data_dir"],
            "/mnt/data/shared/vlm/data",  # covers symlink targets in 10k/all
            CONFIG["video_cache_dir"],
            str(DATA_DIR),
        ],
    )


if __name__ == "__main__":
    main()
