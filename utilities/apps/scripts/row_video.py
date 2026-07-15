"""
Row-own-frames → playable MP4 for Gradio browse apps.

For datasets whose rows are SELF-DESCRIBING about their video (the row carries
`video_frames` (list of frame paths), `fps`/`video_fps`, `need_to_flip`) —
e.g. the VObs-tool-SFT pipeline output (visual_obs/run_tool_sft_4k.py, fields
added 2026-07-15). The row is the source of truth: this module NEVER re-derives
frame paths from session_id/rep_index and NEVER defaults a missing fps —
a missing field returns a loud, distinct status string so a pipeline gap can't
masquerade as a working video ([[feedback_no_silent_fail]], CORE PRINCIPLE of
the pipeline-inspector app: the viewer mirrors the data, it doesn't repair it).

`encode_video()` is lifted from video_sft/app.py (port 7862) — the canonical
fps-correct (`-r fps` container rate) + mirror-correct (hflip) encoder. Kept
byte-compatible in behavior; video_sft still has its own copy (import cycle /
heavy-module concerns) — if you fix a bug here, fix it there too.

Used by: vobs_tool_pipeline/app.py (pipeline-inspector, port 7880).
"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple


def encode_video(image_paths: List[str], fps: float, output_path: str,
                 hflip: bool = False) -> str:
    """Encode frames (webp/png/jpg) to H.264 MP4 for browser playback.
    Lifted from video_sft/app.py::encode_video (2026-07-15)."""
    import shutil
    import subprocess
    import tempfile

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to encode video: {stderr}") from exc
    finally:
        os.unlink(list_file)

    return output_path


def build_row_video(row: dict, cache_dir: str) -> Tuple[Optional[str], str]:
    """Build (or reuse from cache) the playable video for one self-describing row.

    Returns (mp4_path | None, status_message). A None video ALWAYS comes with a
    loud, specific status naming the missing/broken field — never a silent blank.
    Fields read (all row-own, per the producer's 2026-07-15 contract):
      video_frames   list of frame paths — REQUIRED (missing ⇒ pipeline gap)
      video_fps|fps  true per-rep fps — REQUIRED (never defaulted)
      need_to_flip   horizontal-mirror flag — REQUIRED (matches what the
                     teacher saw; missing ⇒ rendered unflipped WITH a warning)
    """
    frames = row.get("video_frames")
    if not isinstance(frames, list) or not frames:
        return None, (
            "🔴 **PIPELINE GAP — `video_frames` not on this row.** "
            "The row is supposed to be self-describing (producer fields added "
            "2026-07-15). This run predates them or the producer regressed — "
            "fix/regenerate at the source (`run_tool_sft_4k.py`), do NOT paper "
            "over it in the viewer."
        )
    missing = [p for p in frames if not os.path.exists(p)]
    if missing:
        return None, (
            f"🔴 **{len(missing)}/{len(frames)} frame files missing on disk** "
            f"(first: `{missing[0]}`). The source rep directory moved or was "
            "cleaned — a data problem, not a viewer problem."
        )

    fps = row.get("video_fps") or row.get("fps")
    if not fps:
        return None, (
            "🔴 **PIPELINE GAP — no `video_fps`/`fps` on this row.** "
            "Refusing to guess a frame rate (wrong fps plays the clip at the "
            "wrong speed — [[feedback_video_fps_and_frames]])."
        )

    flip_val = row.get("need_to_flip")
    flip_warn = ""
    if flip_val is None:
        flip_warn = (" ⚠️ `need_to_flip` missing on row — rendered UNFLIPPED, "
                     "may not match what the teacher saw.")
    hflip = bool(flip_val)

    key_src = "|".join([frames[0], str(len(frames)), f"{float(fps):.5f}", str(hflip)])
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    sess = str(row.get("session_id", "row"))
    rep = str(row.get("rep_index", "x"))
    out = Path(cache_dir) / f"{sess}_{rep}_{key}.mp4"
    if not out.exists():
        encode_video(frames, float(fps), str(out), hflip=hflip)

    status = (
        f"▶ {len(frames)} frames @ {float(fps):.2f} fps · "
        f"{'MIRRORED (need_to_flip=True — as the teacher saw it)' if hflip else 'unmirrored (need_to_flip=False)'}"
        f" · source: `{os.path.dirname(frames[0])}`{flip_warn}"
    )
    return str(out), status
