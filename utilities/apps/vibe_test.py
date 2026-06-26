#!/usr/bin/env python3
"""
vibe_test.py — Free-form VLM inference playground.

Send text, image, or video to any vLLM-served model and inspect the response.
Handles <think>...</think> reasoning traces. Runs on port 7874.

Usage:
    python utilities/apps/vibe_test.py
    # → http://localhost:7874/
"""

import base64
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import gradio as gr
import requests
from openai import OpenAI

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SERVER = "http://localhost:8000"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.7
DEFAULT_THINK = True  # request thinking by default (model-dependent)

# Temp mp4s go to sgsilva tmp, NOT the shared /tmp (output-locations rule).
TMP_DIR = "/mnt/data/sgsilva/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

WORKER_NODES = [f"worker-{i}" for i in range(32)]  # worker-0 … worker-31

# Session dataset root — all sessions (10k + 1805) live under 10k/all
SESSION_ROOT = Path("/mnt/data/shared/vlm/data/10k/all")
VLLM_PORT = 8000
VLLM_PORTS = [8000, 8001, 8002, 8003]  # scan a small range so non-8000 servers show up
SCAN_TIMEOUT = 2.0  # seconds per node

# ── helpers ───────────────────────────────────────────────────────────────────

def _b64(data: bytes, mime: str) -> dict:
    encoded = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _extract_frames(video_path: str, max_frames: int = 64) -> list[dict]:
    """Sample up to max_frames from a video file, return as image_url content blocks."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // max_frames)
    frames = []
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(_b64(buf.tobytes(), "image/jpeg"))
        idx += step
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


# ── session browser helpers ───────────────────────────────────────────────────

def list_sessions(filter_str: str = "") -> list[str]:
    """Return sorted session IDs from SESSION_ROOT, optionally filtered."""
    if not SESSION_ROOT.exists():
        return []
    sessions = sorted(p.name for p in SESSION_ROOT.iterdir() if p.is_dir())
    if filter_str.strip():
        f = filter_str.strip().lower()
        sessions = [s for s in sessions if f in s.lower()]
    return sessions


def list_reps(session_id: str) -> list[str]:
    """Return sorted repetition names for a session."""
    if not session_id:
        return []
    session_dir = SESSION_ROOT / session_id
    for subdir in ("cropped_repetitions", "repetitions"):
        d = session_dir / subdir
        if d.exists():
            reps = sorted(
                p.name for p in d.iterdir()
                if p.is_dir() and p.name.startswith("repetition_")
            )
            return reps
    return []


def _read_fps(session_id: str, rep_name: str) -> float:
    """Read true fps from fps.txt — per-rep first, then session-wide fallback."""
    session_dir = SESSION_ROOT / session_id
    candidates = [
        session_dir / "repetitions" / rep_name / "fps.txt",
        session_dir / "images" / "fps.txt",
    ]
    for p in candidates:
        if p.exists():
            try:
                return float(p.read_text().strip())
            except ValueError:
                pass
    return 25.0  # safe fallback


def build_rep_video(session_id: str, rep_name: str) -> tuple[str | None, str]:
    """
    Build a temporary mp4 from .webp frames for the given rep.
    Returns (tmp_mp4_path, status_message).
    """
    if not session_id or not rep_name:
        return None, "No session/rep selected"

    session_dir = SESSION_ROOT / session_id
    rep_dir = None
    for subdir in ("cropped_repetitions", "repetitions"):
        d = session_dir / subdir / rep_name
        if d.exists():
            rep_dir = d
            break

    if rep_dir is None:
        return None, f"Rep dir not found for {session_id}/{rep_name}"

    frames = sorted(rep_dir.glob("*.webp"))
    if not frames:
        return None, f"No .webp frames in {rep_dir}"

    fps = _read_fps(session_id, rep_name)

    frame_arrays = []
    for fp in frames:
        f = cv2.imread(str(fp))
        if f is not None:
            frame_arrays.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if not frame_arrays:
        return None, f"Could not read any frames from {rep_dir}"

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=TMP_DIR)
    tmp.close()
    import imageio.v3 as iio
    iio.imwrite(tmp.name, frame_arrays, fps=fps, codec="libx264")

    return tmp.name, f"{session_id} / {rep_name} — {len(frame_arrays)} frames @ {fps:.1f} fps"


# ── dataset loader helpers ────────────────────────────────────────────────────

# EXP-B reasoning-trace prompts (the *_gtobs.txt the teacher uses). Loading one
# into the editable Prompt box lets you tweak it live and re-run gemini to see how
# the wording changes the trace — iterate on the prompt without regenerating.
REASONING_VARIANTS_DIR = Path(
    "/home/sgsilva/vlm-post-training/prompts/dataset_creation/add_reasoning_severity_v3_variants")
# The 3 ENFORCED GT-obs prompts (the EXP-B goal: trace must CITE the obs) first,
# then the 3 plain gtobs adaptations. (pmartins' originals omitted — not obs-conditioned.)
# All are {task_section}-style, so they load the same way.
REASONING_VARIANTS = [
    "default_gtobs_ondemand.txt",   # video-primary, consult VObs only on doubt (B-ondemand)
    "default_gtobs_enforced.txt", "more_natural_gtobs_enforced.txt", "without_sections_gtobs_enforced.txt",
    "default_gtobs.txt", "more_natural_gtobs.txt", "without_sections_gtobs.txt",
]


def _strip_prompt_scaffolding(content: str) -> str:
    """Strip the ===/PROMPT/Source/Purpose header + TEMPLATE VARIABLES footer
    documentation so only the instruction body is sent to the teacher (mirrors
    prompt_loader.load_prompt). Falls back to the raw text if no scaffolding."""
    eq = "=" * 80
    m = re.search(rf"{eq}\n.*?{eq}\n\n(.*?)\n\n{eq}\nTEMPLATE VARIABLES:", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    # header but no footer: drop the leading ===…=== doc block if present
    m2 = re.search(rf"^{eq}\n.*?{eq}\n+(.*)$", content, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return content.strip()


def build_reasoning_prompt(dataset_path: str, row_index: int, variant: str) -> tuple[str, str]:
    """Build the EXP-B reasoning-trace prompt for a row: fill the chosen *_gtobs.txt
    {task_section} with the stage-2 question (user turn) + the GT answer as
    <correct_answer>. Returns (prompt, status). Mirrors gen_stage2_severity_traces.py."""
    try:
        ds = load_dataset_cached(dataset_path.strip())
        row = ds[int(row_index)]
        user = row["messages"][1]["content"]
        gt = row["messages"][-1]["content"]
        gt = re.sub(r"<think>.*?</think>\s*", "", gt, flags=re.DOTALL).strip()
        # {task_section} must be DATA ONLY (exercise_description + visual_observations
        # + the response format), NOT the stage-2 user-turn PREAMBLE ("You are an
        # expert physiotherapy assistant…"), which conflicts with the *_gtobs.txt
        # prompt's own "You are a reasoning trace generator…" framing. Strip the
        # preamble: keep from the first <exercise_description> onward.
        idx = user.find("<exercise_description>")
        data = user[idx:] if idx != -1 else user
        task_section = f"{data}\n\n<correct_answer>\n{gt}\n</correct_answer>"
        tmpl = _strip_prompt_scaffolding((REASONING_VARIANTS_DIR / variant).read_text())
        prompt = tmpl.replace("{task_section}", task_section)
        return prompt, f"Loaded reasoning prompt ({variant}) for row {int(row_index)} — edit + Run gemini"
    except Exception as e:
        return "", f"ERROR building reasoning prompt: {type(e).__name__}: {e}"


APP_DATASETS_DIR = Path("/mnt/data/sgsilva/datasets/app_video_datasets")
# Datasets pinned to the TOP of the dropdown. The *_reasoning_sample sets (with
# generated <think> traces) come FIRST — those are the ones to inspect; the plain
# *_enforced/_soft sets are the no-reasoning Phase-1 inputs (empty <think>).
PINNED_DATASETS = [
    "/mnt/data/sgsilva/datasets/app_video_datasets/expb_stage2_reasoning_sample",
    "/mnt/data/sgsilva/datasets/app_video_datasets/expb_stage2_reasoning_sample_soft",
    "/mnt/data/sgsilva/datasets/app_video_datasets/expb_stage2_enforced",
    "/mnt/data/sgsilva/datasets/app_video_datasets/expb_stage2_soft",
]


def _is_hf_dataset(p: Path) -> bool:
    return (p / "dataset_info.json").exists() or (p / "state.json").exists()


def list_app_datasets() -> list[str]:
    """Dropdown choices: pinned EXP-B sets first, then everything else in
    app_video_datasets/ by mtime (newest first). Returns full paths."""
    pinned = [d for d in PINNED_DATASETS if Path(d).exists()]
    others = []
    if APP_DATASETS_DIR.exists():
        cand = [p for p in APP_DATASETS_DIR.iterdir()
                if p.is_dir() and _is_hf_dataset(p) and str(p) not in pinned]
        cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        others = [str(p) for p in cand]
    return pinned + others


_ds_cache: dict = {}  # path → loaded dataset

def load_dataset_cached(path: str):
    if path not in _ds_cache:
        from datasets import load_from_disk
        _ds_cache[path] = load_from_disk(path)
    return _ds_cache[path]


def load_sample(dataset_path: str, row_index: int) -> tuple[str, str, str, object, str]:
    """
    Load one row from an HF Arrow dataset.
    Returns (system_prompt, user_prompt, gt_answer, video_frames_or_None, status).
    """
    if not dataset_path.strip():
        return "", "", "", None, "ERROR: no dataset path"
    try:
        ds = load_dataset_cached(dataset_path.strip())
    except Exception as e:
        return "", "", "", None, f"ERROR loading dataset: {e}"

    n = len(ds)
    if n == 0:
        return "", "", "", None, "ERROR: dataset is empty"
    idx = max(0, min(int(row_index), n - 1))
    row = ds[idx]

    messages = row.get("messages", [])
    if isinstance(messages, str):
        import json
        messages = json.loads(messages)

    system_p, user_p, gt_answer = "", "", ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            system_p = content
        elif role == "user":
            user_p = content
        elif role == "assistant":
            gt_answer = content

    # Decode video frames if present
    video_path = None
    raw_frames = row.get("video_frames")
    if raw_frames:
        try:
            video_path = _frames_to_video(raw_frames, row.get("fps", 25.0))
        except Exception:
            pass

    # Fall back to session video if no embedded frames
    if video_path is None:
        session_id = row.get("session_id") or row.get("video_id", "")
        rep_index = row.get("rep_index")
        if session_id and rep_index is not None:
            rep_name = f"repetition_{rep_index}"
            video_path, _ = build_rep_video(session_id, rep_name)

    need_to_flip = bool(row.get("need_to_flip", True))
    status = f"Row {idx}/{n-1} — session: {row.get('session_id','?')}  rep: {row.get('rep_index','?')}  exercise: {row.get('exercise_id','?')}  flip={need_to_flip}"
    return system_p, user_p, gt_answer, video_path, need_to_flip, status


def _frames_to_video(frames: list, fps: float) -> str | None:
    """Convert a list of frame paths or base64 strings to a temp mp4. Returns path or None."""
    if not frames:
        return None

    def _read_frame(f):
        if isinstance(f, str) and Path(f).exists():
            return cv2.imread(f)
        # base64
        import numpy as np
        try:
            data = base64.b64decode(f) if isinstance(f, str) else f
            return cv2.imdecode(np.frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
        except Exception:
            return None

    frame_arrays = []
    for f in frames:
        frame = _read_frame(f)
        if frame is not None:
            # ensure RGB for imageio
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_arrays.append(frame)
    if not frame_arrays:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=TMP_DIR)
    tmp.close()
    import imageio.v3 as iio
    iio.imwrite(tmp.name, frame_arrays, fps=float(fps), codec="libx264")
    return tmp.name


# ── VO scoring ────────────────────────────────────────────────────────────────

def _parse_obs_lines(text: str) -> dict[int, str]:
    """Parse numbered observation lines from a [VISUAL OBSERVATIONS] block."""
    for marker in ("[VISUAL OBSERVATIONS]", "Visual Observations:"):
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):]
    numbered = re.findall(r"^\s*(\d+)\.\s*(.*?)(?=^\s*\d+\.|\Z)", text,
                          flags=re.DOTALL | re.MULTILINE)
    return {int(n): " ".join(ans.split()) for n, ans in numbered}


def _parse_severity_block(text: str) -> tuple[dict, dict]:
    """Parse a stage-2 bracket answer → ({error_name: severity}, {Effectiveness:n, 'Injury Risk':n}).
    Errors come from the [ERRORS] block; scores from [SCORES]."""
    errs, scores = {}, {}
    eblk = text.split("[ERRORS]", 1)[-1].split("[SCORES]", 1)[0] if "[ERRORS]" in text else ""
    for line in eblk.strip().splitlines():
        if ":" in line:
            k, v = line.rsplit(":", 1)
            m = re.search(r"\d+", v)
            if m:
                errs[k.strip()] = int(m.group())
    sblk = text.split("[SCORES]", 1)[-1].split("[FEEDBACK]", 1)[0] if "[SCORES]" in text else ""
    for label in ("Effectiveness", "Injury Risk"):
        m = re.search(rf"{label}:\s*(\d+)", sblk, re.IGNORECASE)
        if m:
            scores[label] = int(m.group(1))
    return errs, scores


def score_severity(pred: str, gt: str) -> tuple[str, float]:
    """Score a stage-2 [ERRORS]/[SCORES] prediction vs GT: per-error exact-match +
    MAE, plus Effectiveness/Injury exact. Returns (formatted_diff, error_exact_ratio)."""
    gt_err, gt_sc = _parse_severity_block(gt)
    pr_err, pr_sc = _parse_severity_block(pred)
    if not gt_err:
        return "GT has no [ERRORS] block — cannot score", 0.0

    rows, exact, shared_abs, n_shared = [], 0, 0, 0
    for name in gt_err:
        g = gt_err[name]
        if name in pr_err:
            p = pr_err[name]
            n_shared += 1
            d = abs(g - p)
            shared_abs += d
            if d == 0:
                exact += 1
            icon = "✓" if d == 0 else ("≈" if d == 1 else "✗")
            rows.append(f"{icon} {name}: GT {g}  PRED {p}  (Δ{d})")
        else:
            rows.append(f"· {name}: GT {g}  PRED (missing)")
    exact_ratio = exact / len(gt_err)
    mae = (shared_abs / n_shared) if n_shared else None
    sc_rows = []
    for label in ("Effectiveness", "Injury Risk"):
        g, p = gt_sc.get(label), pr_sc.get(label)
        hit = "✓" if (g is not None and g == p) else "✗"
        sc_rows.append(f"{hit} {label}: GT {g}  PRED {p}")

    header = (f"Severity: {exact}/{len(gt_err)} exact ({exact_ratio*100:.0f}%)"
             + (f", err-MAE {mae:.2f}" if mae is not None else "")
             + f", {n_shared}/{len(gt_err)} errors parsed\n"
             + "  ".join(sc_rows) + "\n" + "─"*60 + "\n")
    return header + "\n".join(rows), exact_ratio


def score_vo(pred: str, gt: str) -> tuple[str, float]:
    """
    Score a prediction vs GT. Auto-detects the task by GT format:
      - stage-2 severity ([ERRORS] bracket) → score_severity
      - stage-1 visual-obs (numbered [VISUAL OBSERVATIONS]) → exact-match per line
    Returns (formatted_diff, ratio).
    """
    if "[ERRORS]" in gt:
        return score_severity(pred, gt)

    pred_lines = _parse_obs_lines(pred)
    gt_lines   = _parse_obs_lines(gt)

    if not gt_lines:
        return "GT has no numbered lines or [ERRORS] block — cannot score", 0.0

    all_keys = sorted(set(gt_lines) | set(pred_lines))
    matches = 0
    rows = []
    for k in all_keys:
        g = gt_lines.get(k, "(missing)")
        p = pred_lines.get(k, "(missing)")
        match = (g.lower().strip() == p.lower().strip())
        if match:
            matches += 1
        icon = "✓" if match else "✗"
        rows.append(f"{icon} {k:2d}.  GT  : {g}\n        PRED: {p}")

    ratio = matches / len(gt_lines)
    header = f"Agreement: {matches}/{len(gt_lines)} = {ratio*100:.1f}%\n{'─'*60}\n"
    return header + "\n\n".join(rows), ratio


_flip_cache: dict = {}  # (source_path, source_mtime) -> flipped mp4 path


def _flip_video(video_path: str) -> str | None:
    """Horizontally flip a video → mp4. CACHED by (source path, mtime): re-running
    with only the prompt changed reuses the existing flip instead of rebuilding."""
    try:
        key = (video_path, os.path.getmtime(video_path))
    except OSError:
        key = (video_path, 0)
    cached = _flip_cache.get(key)
    if cached and os.path.exists(cached):
        return cached

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=TMP_DIR)
    tmp.close()
    import imageio.v3 as iio
    iio.imwrite(tmp.name, frames, fps=fps, codec="libx264")
    _flip_cache[key] = tmp.name
    return tmp.name


def _split_think(text: str) -> tuple[str, str]:
    """Return (thinking, answer) split on <think>...</think>."""
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


# ── inference ─────────────────────────────────────────────────────────────────

EVAL_VENV_PY = "/home/sgsilva/vlm-post-training-home-venv/bin/python"
VERTEX_HELPER = str(Path(__file__).parent / "scripts" / "_vertex_call.py")
METRICS_HELPER = str(Path(__file__).parent / "scripts" / "_severity_metrics.py")


def _canonical_metrics(pairs: list) -> str:
    """Run eval.compute_severity_metrics over accumulated (gt, pred) pairs via the
    eval venv (this app's venv lacks sklearn). Returns a formatted block."""
    if not pairs:
        return "(no scored reps yet — run on rows with a stage-2 GT)"
    import subprocess
    try:
        p = subprocess.run([EVAL_VENV_PY, METRICS_HELPER], input=json.dumps(pairs),
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            return f"metrics error: {p.stderr[-300:] or p.stdout[-300:]}"
        m = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        return f"metrics error: {type(e).__name__}: {e}"
    pct = lambda v: f"{v*100:.1f}%" if isinstance(v, (int, float)) else "—"
    g = m.get
    return (
        f"CANONICAL BOARD METRICS over {m.get('n')} scored rep(s):\n"
        f"{'─'*56}\n"
        f"Error-detection : F1 {pct(g('error_detection_f1'))}  "
        f"P {pct(g('error_detection_precision'))}  R {pct(g('error_detection_recall'))}  "
        f"Acc {pct(g('error_detection_accuracy'))}\n"
        f"Sample-level F1 : {pct(g('sample_error_detection_f1'))}\n"
        f"Severity Acc    : exact {pct(g('overall_severity_accuracy'))}  "
        f"within-1 {pct(g('overall_severity_within_1'))}  "
        f"non-1 {pct(g('overall_severity_accuracy_non1'))}\n"
        f"Effectiveness   : exact {pct(g('effectiveness_exact_match_rate'))}  MAE {g('effectiveness_mae')}\n"
        f"Injury Risk     : exact {pct(g('injury_risk_exact_match_rate'))}  MAE {g('injury_risk_mae')}"
    )


def _run_vertex(model_name, prompt, image_file, video_file, system_prompt,
                max_tokens, temperature, max_frames):
    """Route a gemini/Vertex call through the eval venv (this app's venv lacks
    the GCloud SDK). Returns (thinking, answer, status)."""
    import subprocess, os
    req = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt.strip() or None,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    # gemini takes a single video file (not extracted frames); pass the path.
    if video_file is not None:
        req["video_path"] = str(video_file)
    elif image_file is not None:
        req["image_paths"] = [str(image_file)]
    env = dict(os.environ)
    env.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                   "/home/sgsilva/swordhealth-ai-research-af166b5009eb.json")
    env.setdefault("VERTEXAI_PROJECT", "swordhealth-ai-research")
    try:
        p = subprocess.run([EVAL_VENV_PY, VERTEX_HELPER], input=json.dumps(req),
                           capture_output=True, text=True, timeout=300, env=env)
        if p.returncode != 0:
            return "", "", f"ERROR (vertex subprocess): {p.stderr[-400:] or p.stdout[-400:]}"
        out = json.loads(p.stdout.strip().splitlines()[-1])
        return out.get("thinking", ""), out.get("content", ""), out.get("status", "OK")
    except Exception as e:
        return "", "", f"ERROR (vertex): {type(e).__name__}: {e}"


def run_query(
    server_url: str,
    model_name: str,
    prompt: str,
    image_file,       # gr.Image returns a tempfile path (str) or None
    video_file,       # gr.Video returns a tempfile path (str) or None
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    max_frames: int,
) -> tuple[str, str, str]:
    """Returns (thinking_trace, answer, status)."""
    if not prompt.strip():
        return "", "", "ERROR: prompt is empty"
    if not model_name or not str(model_name).strip():
        return "", "", "ERROR: no model selected — scan the cluster and click 'Use selected', or type a model name (e.g. 'gemini-3-flash-preview' for Vertex)"

    # ── Vertex / gemini branch ─────────────────────────────────────────────
    # gemini models run via the eval venv (this app's venv lacks the GCloud SDK);
    # server_url is ignored. Trigger on a gemini/vertex model name.
    _m = str(model_name).strip()
    if "gemini" in _m.lower() or _m.startswith("vertex_ai/"):
        return _run_vertex(_m, prompt, image_file, video_file, system_prompt,
                           max_tokens, temperature, max_frames)

    if not server_url.strip():
        return "", "", "ERROR: server URL is empty"

    # Build content list
    content: list[dict] = []

    if video_file is not None:
        frames = _extract_frames(str(video_file), max_frames=int(max_frames))
        if not frames:
            return "", "", "ERROR: could not extract frames from video"
        content.extend(frames)
        content.append({"type": "text", "text": f"[Video: {len(frames)} frames sampled]\n\n{prompt}"})
    elif image_file is not None:
        img_path = Path(str(image_file))
        mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        content.append(_b64(img_path.read_bytes(), mime))
        content.append({"type": "text", "text": prompt})
    else:
        content.append({"type": "text", "text": prompt})

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": content})

    # thinking budget — Qwen3 style
    extra: dict = {}
    if enable_thinking:
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

    try:
        client = OpenAI(base_url=server_url.rstrip("/") + "/v1", api_key="EMPTY")
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            **extra,
        )
    except Exception as e:
        return "", "", f"ERROR: {e}"

    choice = resp.choices[0]
    raw = choice.message.content or ""
    finish_reason = choice.finish_reason or "?"

    # Check reasoning_content field (vLLM separates thinking here when enable_thinking=True)
    thinking_from_field = ""
    try:
        thinking_from_field = choice.message.reasoning_content or ""
    except AttributeError:
        pass

    if thinking_from_field:
        thinking = thinking_from_field.strip()
        answer = raw.strip()
    else:
        thinking, answer = _split_think(raw)

    tokens_in = resp.usage.prompt_tokens if resp.usage else "?"
    tokens_out = resp.usage.completion_tokens if resp.usage else "?"
    status = f"OK — {tokens_in} prompt / {tokens_out} completion tokens — finish: {finish_reason}"

    # Warn if answer is empty but thinking is not (model exhausted token budget thinking)
    if not answer and thinking:
        answer = "[No answer — model used all tokens in <think>. Try disabling thinking or increasing max_tokens.]"

    return thinking, answer, status


# ── fetch model list from server ───────────────────────────────────────────────

def fetch_models(server_url: str) -> tuple[list[str], str]:
    try:
        client = OpenAI(base_url=server_url.rstrip("/") + "/v1", api_key="EMPTY")
        models = [m.id for m in client.models.list()]
        if not models:
            return [], "No models found at this server"
        return models, f"Found {len(models)} model(s)"
    except Exception as e:
        return [], f"ERROR: {e}"


# Vertex/gemini models — always selectable in the dropdown (no server needed).
GEMINI_MODELS = ["gemini-3-flash-preview", "gemini-3.1-pro-preview"]


def refresh_models(server_url: str):
    models, status = fetch_models(server_url)
    # Always offer the gemini models too — they route via Vertex, no server.
    choices = list(GEMINI_MODELS) + [m for m in models if m not in GEMINI_MODELS]
    value = (models[0] if models else GEMINI_MODELS[0])
    if not models:
        status = (status or "") + "  (server has no models — gemini still available)"
    return gr.update(choices=choices, value=value), status


# ── cluster scan ──────────────────────────────────────────────────────────────

def _get_vllm_owner(node: str, port: int = VLLM_PORT) -> str:
    """SSH to node and return the user OWNING the server on this PORT.

    A node can host several vLLM servers on different ports (e.g. worker-30 with
    jmendon on :8000 and sgsilva on :8001), so resolve the owner of the process
    actually LISTENING on `port` — not just the first vllm process on the node.
    """
    import subprocess
    # Find the PID listening on :port (ss → PID), then its user (ps -o user=).
    # Fall back to a node-wide vllm grep only if the port lookup yields nothing.
    cmd = (
        f"pid=$(ss -ltnp 2>/dev/null | grep ':{port} ' "
        f"| sed -n 's/.*pid=\\([0-9]*\\).*/\\1/p' | head -1); "
        f"u=$([ -n \"$pid\" ] && ps -o user= -p \"$pid\" 2>/dev/null | tr -d ' '); "
        f"if [ -n \"$u\" ]; then echo \"$u\"; "
        f"else ps aux | grep vllm | grep -v grep | awk '{{print $1}}' | head -1; fi"
    )
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes", node, cmd],
            capture_output=True, text=True, timeout=5,
        )
        owner = result.stdout.strip()
        return owner if owner else "?"
    except Exception:
        return "?"


def _probe_node(node: str, port: int = VLLM_PORT) -> tuple[str, int, list[str], str] | None:
    """Return (node, port, [model_ids], owner) if a vLLM server is live, else None."""
    url = f"http://{node}:{port}/v1/models"
    try:
        r = requests.get(url, timeout=SCAN_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            models = [m["id"] for m in data.get("data", [])]
            owner = _get_vllm_owner(node, port)
            return node, port, models, owner
    except Exception:
        pass
    return None


def scan_cluster() -> tuple[str, list, list[str]]:
    """
    Probe all worker nodes in parallel.
    Returns (scan_summary_text, results, choices).
    results: list of (node, [model_ids], owner)
    choices: "worker-N | owner | <model name>" — shown in dropdown
    """
    results = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {pool.submit(_probe_node, node, port): (node, port)
                   for node in WORKER_NODES for port in VLLM_PORTS}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    if not results:
        return "No vLLM servers found on worker-0 … worker-31", [], []

    results.sort(key=lambda x: (int(x[0].split("-")[1]), x[1]))

    lines = []
    choices = []
    for node, port, models, owner in results:
        # only annotate the port when it's not the default, to keep labels clean
        port_tag = "" if port == VLLM_PORT else f":{port}"
        for mid in models:
            short = mid.split("/")[-1]  # short name for display
            label = f"{node}{port_tag} | {owner} | {short}"
            choices.append(label)
            lines.append(f"✓ {node}:{port}  [{owner}]  {short}")

    summary = f"Found {len(results)} live server(s):\n" + "\n".join(lines)
    return summary, results, choices


def apply_scan_selection(selected: str, scan_results: list) -> tuple[str, str]:
    """Given a selection 'worker-N[:PORT] | owner | short_name', return (server_url, full_model_id)."""
    if not selected or not scan_results:
        return "", ""
    parts = [p.strip() for p in selected.split("|")]
    if len(parts) < 3:
        return "", ""
    node_part = parts[0]            # "worker-N" or "worker-N:PORT"
    short = parts[2]
    if ":" in node_part:
        sel_node, sel_port = node_part.split(":", 1)
        sel_port = int(sel_port)
    else:
        sel_node, sel_port = node_part, VLLM_PORT
    for node, port, models, owner in scan_results:
        if node == sel_node and port == sel_port:
            for mid in models:
                if mid.split("/")[-1] == short:
                    return f"http://{node}:{port}", mid
            return f"http://{node}:{port}", models[0] if models else ""
    return "", ""


# ── activity log ──────────────────────────────────────────────────────────────

import datetime

def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def _append(log: list, *lines: str) -> list:
    new = list(log)
    for line in lines:
        new.append(f"[{_ts()}] {line}")
    return new[-200:]  # keep last 200 lines

def _render(log: list) -> str:
    return "\n".join(log)


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="VLM Vibe Tester") as demo:
        # state: list of (node, [model_ids]) from last scan — must live inside Blocks
        scan_state = gr.State([])
        activity_log = gr.State([])
        metrics_pairs = gr.State([])   # accumulated [{gt, pred}] for canonical metrics

        gr.Markdown("# VLM Vibe Tester\nSend text / image / video to any vLLM-served model and inspect the output.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Server")

                with gr.Group():
                    gr.Markdown("**Scan cluster** — find all live vLLM servers on worker-0…31")
                    scan_btn = gr.Button("Scan cluster", size="sm", variant="secondary")
                    scan_summary = gr.Textbox(label="", show_label=False, lines=4,
                                              interactive=False, placeholder="Click Scan to discover live servers…")
                    scan_picker = gr.Dropdown(label="Pick a server + model", choices=[],
                                              value="", allow_custom_value=True)
                    use_btn = gr.Button("Use selected", size="sm")

                gr.Markdown("— or enter manually —")
                server_url = gr.Textbox(
                    label="Server URL",
                    value=DEFAULT_SERVER,
                    placeholder="http://worker-3:8000",
                )
                model_name = gr.Dropdown(
                    label="Model name (or path)",
                    choices=list(GEMINI_MODELS),   # gemini selectable on load (no server needed)
                    value=GEMINI_MODELS[0],
                    allow_custom_value=True,
                    info="Gemini models route via Vertex (Server URL ignored). "
                         "Scan the cluster + Refresh to add served vLLM models.",
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh models", size="sm")
                    server_status = gr.Textbox(label="", show_label=False, interactive=False, scale=3)

                submit_btn = gr.Button("Run", variant="primary")

                gr.Markdown("### Input")

                with gr.Group():
                    gr.Markdown("**Load from dataset** — pre-fills prompt + video from an HF Arrow dataset row")
                    with gr.Row():
                        dataset_dd = gr.Dropdown(
                            label="Pick a dataset (app_video_datasets/ — EXP-B on top)",
                            choices=list_app_datasets(),
                            value=(list_app_datasets()[0] if list_app_datasets() else None),
                            scale=5, allow_custom_value=True,
                        )
                        ds_refresh_btn = gr.Button("↻", size="sm", scale=1)
                        app_refresh_btn = gr.Button("🔄 Refresh app", size="sm", scale=2)
                    dataset_path = gr.Textbox(
                        label="Dataset path (or type any HF Arrow path)",
                        value=(list_app_datasets()[0] if list_app_datasets() else ""),
                        placeholder="/mnt/data/shared/vlm/data/...",
                        lines=1,
                    )
                    with gr.Row():
                        row_index = gr.Number(label="Row index", value=0, precision=0)
                        load_sample_btn = gr.Button("Load sample", size="sm")
                    with gr.Row():
                        reasoning_variant_dd = gr.Dropdown(
                            label="Reasoning prompt variant",
                            choices=REASONING_VARIANTS, value=REASONING_VARIANTS[0], scale=3)
                        load_reasoning_prompt_btn = gr.Button(
                            "Load reasoning prompt → Prompt box", size="sm", scale=2)
                    gt_answer = gr.Textbox(label="Ground-truth answer (from dataset)", lines=4, interactive=False)
                    dataset_status = gr.Textbox(label="", show_label=False, interactive=False)

                prompt = gr.Textbox(
                    label="Prompt",
                    lines=6,
                    placeholder="Describe what you see in this video.",
                )
                system_prompt = gr.Textbox(
                    label="System prompt (optional)",
                    lines=2,
                    value="You are an AI physical therapist within the Thrive program by Sword Health.",
                )
                image_input = gr.Image(
                    label="Image (optional)",
                    type="filepath",
                    sources=["upload"],
                )

                with gr.Group():
                    gr.Markdown("**Video — browse sessions** (10k/all)")
                    with gr.Row():
                        session_filter = gr.Textbox(
                            label="Filter sessions (exercise code or ID)",
                            placeholder="e.g. 10052",
                            scale=3,
                        )
                        session_search_btn = gr.Button("Search", size="sm")
                    session_picker = gr.Dropdown(
                        label="Session", choices=[], value="", allow_custom_value=False,
                    )
                    rep_picker = gr.Dropdown(
                        label="Repetition", choices=[], value="", allow_custom_value=False,
                    )
                    load_video_btn = gr.Button("Load rep as video", size="sm")
                    session_status = gr.Textbox(label="", show_label=False, interactive=False)

                with gr.Row():
                    video_input = gr.Video(
                        label="Video (loaded from session or upload)",
                        sources=["upload"],
                    )
                flip_video = gr.Checkbox(value=True, label="Flip video horizontally (need_to_flip)")

                gr.Markdown("### Params")
                with gr.Row():
                    max_tokens = gr.Slider(256, 32768, value=DEFAULT_MAX_TOKENS, step=256, label="Max tokens")
                    temperature = gr.Slider(0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
                with gr.Row():
                    enable_thinking = gr.Checkbox(value=DEFAULT_THINK, label="Enable thinking (Qwen3 / budget_tokens)")
                    max_frames = gr.Slider(8, 128, value=64, step=8, label="Max video frames")


            with gr.Column(scale=1):
                gr.Markdown("### Output")
                run_status = gr.Textbox(label="Status", interactive=False)
                answer_out = gr.Textbox(label="Answer", lines=12, interactive=False)
                score_out = gr.Textbox(label="Score vs GT (this rep)", lines=8, interactive=False)
                metrics_out = gr.Textbox(
                    label="Canonical board metrics (accumulates across runs)",
                    lines=8, interactive=False,
                    value="(no scored reps yet — run on rows with a stage-2 GT)")
                metrics_reset_btn = gr.Button("Reset accumulated metrics", size="sm")
                thinking_out = gr.Textbox(label="Thinking trace", lines=12, interactive=False)

        gr.Markdown("---")
        log_box = gr.Textbox(
            label="Activity log",
            lines=12,
            max_lines=12,
            interactive=False,
            autoscroll=True,
            placeholder="Actions, model calls, warnings and errors appear here…",
        )

        # wiring
        def _do_scan(log):
            log = _append(log, "→ Scanning worker-0…31 for live vLLM servers…")
            summary, results, choices = scan_cluster()
            log = _append(log, summary.replace("\n", " | "))
            return summary, gr.update(choices=choices, value=choices[0] if choices else ""), results, _render(log), log

        scan_btn.click(
            fn=_do_scan,
            inputs=[activity_log],
            outputs=[scan_summary, scan_picker, scan_state, log_box, activity_log],
        )

        def _use_selected(selected, results, log):
            url, mid = apply_scan_selection(selected, results)
            status = f"Using {url}  model: {mid}" if mid else "ERROR: nothing selected"
            log = _append(log, f"→ Selected: {selected}", f"  server={url}  model={mid}")
            # keep gemini selectable alongside the picked served model
            choices = ([mid] if mid else []) + list(GEMINI_MODELS)
            return url, gr.update(choices=choices, value=mid or GEMINI_MODELS[0]), status, _render(log), log

        use_btn.click(
            fn=_use_selected,
            inputs=[scan_picker, scan_state, activity_log],
            outputs=[server_url, model_name, server_status, log_box, activity_log],
        )

        def _search_sessions(f, log):
            sessions = list_sessions(f)
            msg = f"Found {len(sessions)} sessions matching '{f}'"
            log = _append(log, f"→ Session search: {msg}")
            return gr.update(choices=sessions, value=sessions[0] if sessions else ""), msg, _render(log), log

        def _on_session_pick(session_id):
            reps = list_reps(session_id)
            return gr.update(choices=reps, value=reps[0] if reps else "")

        def _load_rep_video(session_id, rep_name, log):
            log = _append(log, f"→ Loading video: {session_id} / {rep_name}")
            path, status = build_rep_video(session_id, rep_name)
            log = _append(log, f"  {status}")
            return path, status, _render(log), log

        session_search_btn.click(
            fn=_search_sessions,
            inputs=[session_filter, activity_log],
            outputs=[session_picker, session_status, log_box, activity_log],
        )
        session_picker.change(
            fn=_on_session_pick,
            inputs=[session_picker],
            outputs=[rep_picker],
        )
        load_video_btn.click(
            fn=_load_rep_video,
            inputs=[session_picker, rep_picker, activity_log],
            outputs=[video_input, session_status, log_box, activity_log],
        )

        DEFAULT_SYS = "You are an AI physical therapist within the Thrive program by Sword Health."

        def _load_sample(path, idx, log):
            log = _append(log, f"→ Loading dataset row {int(idx)} from {Path(path).name if path else '?'}")
            sys_p, user_p, gt, video_path, need_flip, status = load_sample(path, int(idx))
            log = _append(log, f"  {status}")
            if video_path:
                log = _append(log, f"  video built: {video_path}")
            # If the assistant turn carries a STORED reasoning trace (generated
            # dataset: <think>…non-empty…</think>), split it so the trace shows in
            # the Thinking box and only the answer stays in the GT box.
            stored_think = ""
            t, a = _split_think(gt)
            if t:
                stored_think = t
                gt = a
                log = _append(log, f"  stored <think> trace: {len(t)} chars")
            return (sys_p or DEFAULT_SYS, user_p, gt, video_path, need_flip,
                    stored_think, status, _render(log), log)

        load_sample_btn.click(
            fn=_load_sample,
            inputs=[dataset_path, row_index, activity_log],
            outputs=[system_prompt, prompt, gt_answer, video_input, flip_video,
                     thinking_out, dataset_status, log_box, activity_log],
        )

        # "Load reasoning prompt": fill the editable Prompt box with the *_gtobs.txt
        # reasoning prompt for this row + load the video, so you can tweak the prompt
        # and Run gemini-flash to see the resulting trace. Also blanks the system box
        # (the reasoning prompt is self-contained).
        def _load_reasoning_prompt(path, idx, variant, log):
            prompt_text, status = build_reasoning_prompt(path, int(idx), variant)
            log = _append(log, f"→ {status}")
            _sp, _up, _gt, video_path, need_flip, _st = load_sample(path, int(idx))
            return "", prompt_text, video_path, need_flip, status, _render(log), log

        load_reasoning_prompt_btn.click(
            fn=_load_reasoning_prompt,
            inputs=[dataset_path, row_index, reasoning_variant_dd, activity_log],
            outputs=[system_prompt, prompt, video_input, flip_video,
                     dataset_status, log_box, activity_log],
        )

        # dataset dropdown → fill the path box; refresh ↻ → re-scan app_video_datasets/
        dataset_dd.change(fn=lambda p: p or "", inputs=[dataset_dd], outputs=[dataset_path])
        ds_refresh_btn.click(
            fn=lambda: gr.update(choices=list_app_datasets(),
                                 value=(list_app_datasets()[0] if list_app_datasets() else None)),
            inputs=[], outputs=[dataset_dd],
        )

        # 🔄 Refresh app: CLEAR the dataset cache (so an overwritten dataset reloads
        # fresh, not the stale cached copy) + re-scan the dropdown, then reload the
        # current row.
        app_refresh_btn.click(
            fn=lambda: (_ds_cache.clear(), gr.update(choices=list_app_datasets()))[1],
            inputs=[], outputs=[dataset_dd],
        ).then(
            fn=_load_sample,
            inputs=[dataset_path, row_index, activity_log],
            outputs=[system_prompt, prompt, gt_answer, video_input, flip_video,
                     thinking_out, dataset_status, log_box, activity_log],
        )

        def _refresh_models(url, log):
            log = _append(log, f"→ Refreshing models from {url}")
            update, status = refresh_models(url)
            log = _append(log, f"  {status}")
            return update, status, _render(log), log

        refresh_btn.click(
            fn=_refresh_models,
            inputs=[server_url, activity_log],
            outputs=[model_name, server_status, log_box, activity_log],
        )

        def _run_query(server_url, model_name, prompt, image_file, video_file,
                       system_prompt, max_tokens, temperature, enable_thinking,
                       max_frames, flip, gt, log, pairs):
            # Apply horizontal flip if requested
            flipped_video = None
            if video_file and flip:
                flipped_video = _flip_video(video_file)
            effective_video = flipped_video or video_file

            modality = "video" if effective_video else ("image" if image_file else "text")
            log = _append(log,
                f"→ RUN  model={model_name}  server={server_url}",
                f"   modality={modality}  max_tokens={max_tokens}  thinking={'on' if enable_thinking else 'off'}  flip={flip}",
                f"   prompt={prompt[:80].strip()!r}{'…' if len(prompt)>80 else ''}",
            )
            thinking, answer, status = run_query(
                server_url, model_name, prompt, image_file, effective_video,
                system_prompt, max_tokens, temperature, enable_thinking, max_frames,
            )
            pairs = list(pairs or [])
            if status.startswith("ERROR"):
                log = _append(log, f"  ✗ {status}")
                return thinking, answer, "", _canonical_metrics(pairs), status, _render(log), log, pairs

            log = _append(log, f"  ✓ {status}")
            if answer:
                log = _append(log, f"  answer={answer[:120].strip()!r}{'…' if len(answer)>120 else ''}")
            if thinking:
                log = _append(log, f"  think={len(thinking)} chars")

            # Score against GT if available
            score_text = ""
            if gt and gt.strip() and answer:
                score_text, ratio = score_vo(answer, gt)
                log = _append(log, f"  score: {ratio*100:.1f}%")
                # accumulate stage-2 (gt, pred) for canonical pooled metrics
                if "[ERRORS]" in gt:
                    pairs.append({"gt": gt, "pred": answer})

            metrics_text = _canonical_metrics(pairs)
            return thinking, answer, score_text, metrics_text, status, _render(log), log, pairs

        submit_btn.click(
            fn=_run_query,
            inputs=[
                server_url, model_name, prompt,
                image_input, video_input,
                system_prompt, max_tokens, temperature,
                enable_thinking, max_frames, flip_video, gt_answer, activity_log, metrics_pairs,
            ],
            outputs=[thinking_out, answer_out, score_out, metrics_out, run_status, log_box, activity_log, metrics_pairs],
        )

        metrics_reset_btn.click(
            fn=lambda: ([], "(reset — no scored reps)"),
            inputs=[], outputs=[metrics_pairs, metrics_out],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7874)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=False, theme=gr.themes.Soft())
