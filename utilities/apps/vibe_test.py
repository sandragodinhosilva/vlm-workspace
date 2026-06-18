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

WORKER_NODES = [f"worker-{i}" for i in range(32)]  # worker-0 … worker-31

# Session dataset root — all sessions (10k + 1805) live under 10k/all
SESSION_ROOT = Path("/mnt/data/shared/vlm/data/10k/all")
VLLM_PORT = 8000
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

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp")
    tmp.close()
    import imageio.v3 as iio
    iio.imwrite(tmp.name, frame_arrays, fps=fps, codec="libx264")

    return tmp.name, f"{session_id} / {rep_name} — {len(frame_arrays)} frames @ {fps:.1f} fps"


# ── dataset loader helpers ────────────────────────────────────────────────────

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

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp")
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


def score_vo(pred: str, gt: str) -> tuple[str, float]:
    """
    Score a VO prediction against GT by exact-match per numbered line.
    Returns (formatted_diff, agreement_ratio).
    """
    pred_lines = _parse_obs_lines(pred)
    gt_lines   = _parse_obs_lines(gt)

    if not gt_lines:
        return "GT has no numbered lines — cannot score", 0.0

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


def _flip_video(video_path: str) -> str | None:
    """Horizontally flip all frames of a video, return path to new /tmp mp4."""
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
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp")
    tmp.close()
    import imageio.v3 as iio
    iio.imwrite(tmp.name, frames, fps=fps, codec="libx264")
    return tmp.name


def _split_think(text: str) -> tuple[str, str]:
    """Return (thinking, answer) split on <think>...</think>."""
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


# ── inference ─────────────────────────────────────────────────────────────────

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
    if not server_url.strip():
        return "", "", "ERROR: server URL is empty"
    if not model_name or not str(model_name).strip():
        return "", "", "ERROR: no model selected — scan the cluster and click 'Use selected', or type a model name manually"

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


def refresh_models(server_url: str):
    models, status = fetch_models(server_url)
    choices = models if models else []
    value = choices[0] if choices else ""
    return gr.update(choices=choices, value=value), status


# ── cluster scan ──────────────────────────────────────────────────────────────

def _get_vllm_owner(node: str) -> str:
    """SSH to node and return the user running vLLM, or '?' on failure."""
    import subprocess
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes", node,
             "ps aux | grep vllm | grep -v grep | awk '{print $1}' | head -1"],
            capture_output=True, text=True, timeout=4,
        )
        owner = result.stdout.strip()
        return owner if owner else "?"
    except Exception:
        return "?"


def _probe_node(node: str) -> tuple[str, list[str], str] | None:
    """Return (node, [model_ids], owner) if a vLLM server is live on that node, else None."""
    url = f"http://{node}:{VLLM_PORT}/v1/models"
    try:
        r = requests.get(url, timeout=SCAN_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            models = [m["id"] for m in data.get("data", [])]
            owner = _get_vllm_owner(node)
            return node, models, owner
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
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_probe_node, node): node for node in WORKER_NODES}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    if not results:
        return "No vLLM servers found on worker-0 … worker-31", [], []

    results.sort(key=lambda x: int(x[0].split("-")[1]))

    lines = []
    choices = []
    for node, models, owner in results:
        url = f"http://{node}:{VLLM_PORT}"
        for mid in models:
            short = mid.split("/")[-1]  # short name for display
            label = f"{node} | {owner} | {short}"
            choices.append(label)
            lines.append(f"✓ {node}  [{owner}]  {short}")

    summary = f"Found {len(results)} live server(s):\n" + "\n".join(lines)
    return summary, results, choices


def apply_scan_selection(selected: str, scan_results: list) -> tuple[str, str]:
    """Given a selection 'worker-N | owner | short_name', return (server_url, full_model_id)."""
    if not selected or not scan_results:
        return "", ""
    parts = [p.strip() for p in selected.split("|")]
    if len(parts) < 3:
        return "", ""
    node_part = parts[0]
    short = parts[2]
    for node, models, owner in scan_results:
        if node == node_part:
            for mid in models:
                if mid.split("/")[-1] == short:
                    return f"http://{node}:{VLLM_PORT}", mid
            return f"http://{node}:{VLLM_PORT}", models[0] if models else ""
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
                    choices=[],
                    value="",
                    allow_custom_value=True,
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh models", size="sm")
                    server_status = gr.Textbox(label="", show_label=False, interactive=False, scale=3)

                submit_btn = gr.Button("Run", variant="primary")

                gr.Markdown("### Input")

                with gr.Group():
                    gr.Markdown("**Load from dataset** — pre-fills prompt + video from an HF Arrow dataset row")
                    dataset_path = gr.Textbox(
                        label="Dataset path",
                        placeholder="/mnt/data/shared/vlm/data/human_annotation_datasets/1805_not_reviewed_visual_obs/1805_oracle_obs_sft_train_categorical",
                        lines=1,
                    )
                    with gr.Row():
                        row_index = gr.Number(label="Row index", value=0, precision=0)
                        load_sample_btn = gr.Button("Load sample", size="sm")
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
                score_out = gr.Textbox(label="VO score vs GT (auto when GT is loaded)", lines=10, interactive=False)
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
            return url, gr.update(choices=[mid] if mid else [], value=mid), status, _render(log), log

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
            return sys_p or DEFAULT_SYS, user_p, gt, video_path, need_flip, status, _render(log), log

        load_sample_btn.click(
            fn=_load_sample,
            inputs=[dataset_path, row_index, activity_log],
            outputs=[system_prompt, prompt, gt_answer, video_input, flip_video, dataset_status, log_box, activity_log],
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
                       max_frames, flip, gt, log):
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
            if status.startswith("ERROR"):
                log = _append(log, f"  ✗ {status}")
                return thinking, answer, "", status, _render(log), log

            log = _append(log, f"  ✓ {status}")
            if answer:
                log = _append(log, f"  answer={answer[:120].strip()!r}{'…' if len(answer)>120 else ''}")
            if thinking:
                log = _append(log, f"  think={len(thinking)} chars")

            # Score against GT if available
            score_text = ""
            if gt and gt.strip() and answer:
                score_text, ratio = score_vo(answer, gt)
                log = _append(log, f"  VO score: {ratio*100:.1f}%")

            return thinking, answer, score_text, status, _render(log), log

        submit_btn.click(
            fn=_run_query,
            inputs=[
                server_url, model_name, prompt,
                image_input, video_input,
                system_prompt, max_tokens, temperature,
                enable_thinking, max_frames, flip_video, gt_answer, activity_log,
            ],
            outputs=[thinking_out, answer_out, score_out, run_status, log_box, activity_log],
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
