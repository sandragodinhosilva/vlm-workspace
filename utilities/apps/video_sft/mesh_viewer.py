#!/usr/bin/env python3
"""3D Mesh Viewer — browse SAM-3D-Body overlay videos for any (video, rep).

Combines:
  - app_mesh_render.py: rasterized overlay mp4s (mesh, 3d_skel, combined)
  - sword_viewer.py:    interactive Plotly 3D skeleton + mesh + frame slider
  - geometry_3d.py:     per-frame angle signals + time-series chart

Usage:
    cd /home/sgsilva/utilities/apps/video_sft
    source /home/sgsilva/vlm-post-training-home-venv/bin/activate
    lsof -ti:7863 | xargs -r kill -9
    python mesh_viewer.py --port 7863
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

# Reuse frame helpers + SAM3D helpers from app.py
from app import (  # noqa: E402
    CONFIG,
    _rep_filenames,
    _resolve_filenames_to_source,
    _sam3d_output_dir,
    _sam3d_searched_dirs,
    _resolve_video_for_mode,
    _load_mesh_render_stats,
    get_gallery_frames,
    get_or_create_video,
    get_video_fps,
    encode_video,
    resolve_video_dir,
    _iter_rep_indices,
    _VIDEO_FLIP_CACHE,
    _3D_MODES,
)
from app_mesh_render import RenderStats  # noqa: E402

# geometry_3d lives in the sam3d_pilot dir
_PILOT_DIR = "/home/sgsilva/vlm-post-training/aux_tasks/video_tasks/video_mcqa/sam3d_pilot"
if _PILOT_DIR not in sys.path:
    sys.path.insert(0, _PILOT_DIR)
from geometry_3d import compute_all_signals  # noqa: E402

# ---------------------------------------------------------------------------
# MHR-70 joint taxonomy (from sword_viewer.py / geometry_3d.py)
# ---------------------------------------------------------------------------
COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO17_BONES = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
# MHR-70 → COCO-17 remapping (source: metadata/__init__.py MHR70_TO_COCO17, inverted)
COCO17_TO_MHR70 = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14], dtype=np.int64
)

_SIGNAL_COLORS = {
    "trunk_axial_yaw":    "#1f77b4",
    "head_yaw":           "#ff7f0e",
    "trunk_angle":        "#888888",
    "left_hip_abduction": "#2ca02c",
    "right_hip_abduction":"#d62728",
    "left_knee_flexion":  "#9467bd",
    "right_knee_flexion": "#8c564b",
}

# ---------------------------------------------------------------------------
# Known SAM3D-covered videos
# ---------------------------------------------------------------------------
_KNOWN_VIDEOS = [
    # Fresh rep-structured SAM-3D run 2026-06-30 (16 reps, extras in
    # cropped_repetitions_3d/repetition_N/) — the live smoke-test session.
    ("10001_833874_16122025104010_64248596", "2504 fresh (seated)",
     "/mnt/data/shared/vlm/data/human_annotations/2504_processed"),
    ("10073_515668_24102025213225_58864551", "seated trunk rotation",
     "/mnt/data/sgsilva/tmp/post_revert/10073"),
    ("12002_198332_24102025063751_29945545", "standing",
     "/mnt/data/sgsilva/tmp/post_revert/12002"),
    ("13003_511825_03112025210349_41076008", "seated",
     "/mnt/data/sgsilva/tmp/post_revert/13003"),
    ("10052_258964_11112025131111_24778272", "supine / Bridge",
     "/mnt/data/sgsilva/results/sam3dbody_audit/pose_class_audit/supine"),
    ("12202_361586_18102025163927_35255761", "prone / Press Up",
     "/mnt/data/sgsilva/results/sam3dbody_audit/pose_class_audit/prone"),
    ("10055_236215_06112025062738_23857806", "side_lying / Hip Abduction",
     "/mnt/data/sgsilva/results/sam3dbody_audit/pose_class_audit/side_lying"),
    ("12200_326077_07112025111823_30455810", "kneeling / Birddog",
     "/mnt/data/sgsilva/results/sam3dbody_audit/pose_class_audit/kneeling"),
]

# Register each known video's OWN root, keyed by video_id, so the resolver only
# ever returns a video's own extras (a shared audit dir can't leak across videos).
CONFIG.setdefault("sam3d_video_roots", {})
CONFIG.setdefault("extra_data_roots", [])
for _vid, _label, _root in _KNOWN_VIDEOS:
    CONFIG["sam3d_video_roots"][_vid] = _root
    # If the root is a dataset tree that CONTAINS <video_id>/ as a session subdir
    # (e.g. a _processed cohort), register it so resolve_video_dir finds the bg
    # frames for the overlay video. Flat audit dirs (root == the video's own dir)
    # are skipped here — they have no <root>/<video_id> child.
    if (Path(_root) / _vid).is_dir() and _root not in CONFIG["extra_data_roots"]:
        CONFIG["extra_data_roots"].append(_root)

_VIDEO_CHOICES = [(f"{label}  [{vid[:5]}…]", vid) for vid, label, _ in _KNOWN_VIDEOS]

_MODE_CHOICES = [
    ("Raw frames",              "raw"),
    ("2D skeleton",             "skeleton"),
    ("Mesh overlay",            "mesh"),
    ("3D skeleton",             "3d_skel"),
    ("Mesh + 3D skeleton",      "mesh_kp_combined"),
    ("Side-by-side: raw|mesh",  "side_by_side_raw_mesh"),
    ("Side-by-side: mesh|skel", "side_by_side_mesh_skel"),
]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _rep_choices(video_id: str) -> List[str]:
    if not video_id:
        return []
    try:
        idxs = _iter_rep_indices(video_id)
    except Exception:
        idxs = []
    return [str(i) for i in idxs] if idxs else ["0"]


def _load_pose3d_extras(video_id: str, rep_index: int) -> Optional[Dict]:
    """Load _3d_extras.json for a (video_id, rep) pair and return parsed data dict."""
    sam3d_dir = _sam3d_output_dir(video_id, rep_index)
    if sam3d_dir is None:
        return None
    hits = list(sam3d_dir.glob("*_3d_extras.json"))
    if not hits:
        return None
    try:
        extras = json.loads(hits[0].read_text())
    except Exception:
        return None

    by_id = {f["image_id"]: f for f in extras.get("frames", [])}
    image_ids = sorted(by_id.keys())

    # Load mesh npz if present
    meshes = None
    mesh_hits = list(sam3d_dir.glob("*_3d_meshes.npz"))
    if mesh_hits:
        try:
            npz = np.load(str(mesh_hits[0]))
            meshes = {
                "vertices": npz["vertices"],
                "image_ids": npz["image_ids"],
                "faces": npz["faces"] if "faces" in npz.files else None,
            }
        except Exception:
            pass

    # Compute angle signals across all frames
    signals = compute_all_signals(extras.get("frames", []))

    return {
        "extras": extras,
        "by_id": by_id,
        "image_ids": image_ids,
        "meshes": meshes,
        "signals": signals,
        "sam3d_dir": sam3d_dir,
    }


# ---------------------------------------------------------------------------
# Panel renderers (from sword_viewer.py, adapted)
# ---------------------------------------------------------------------------

def _render_3d_skeleton_plotly(data: Dict, frame_idx: int) -> go.Figure:
    image_ids = data["image_ids"]
    if not (0 <= frame_idx < len(image_ids)):
        return go.Figure().update_layout(title="frame out of range")
    fm = data["by_id"][image_ids[frame_idx]]
    kp3d = np.asarray(fm["pred_keypoints_3d"])
    cam_t = np.asarray(fm.get("pred_cam_t", [0, 0, 0]))
    pts_all = kp3d + cam_t[None, :]
    pts = pts_all[COCO17_TO_MHR70]  # (17,3) COCO-17 order

    fig = go.Figure()
    name_to_idx = {n: i for i, n in enumerate(COCO17_NAMES)}
    for a, b in COCO17_BONES:
        ai, bi = name_to_idx[a], name_to_idx[b]
        pa, pb = pts[ai], pts[bi]
        fig.add_trace(go.Scatter3d(
            x=[pa[0], pb[0]], y=[pa[1], pb[1]], z=[pa[2], pb[2]],
            mode="lines", line=dict(color="cyan", width=4),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers", marker=dict(size=5, color="red"),
        text=COCO17_NAMES, hoverinfo="text", showlegend=False,
    ))
    cx, cy, cz = float(pts[:, 0].mean()), float(pts[:, 1].mean()), float(pts[:, 2].mean())
    fig.update_layout(
        title=f"3D skeleton — frame {frame_idx + 1}/{len(image_ids)} (drag to rotate)",
        scene=dict(
            xaxis_title="X (cam-right)", yaxis_title="Y (cam-down)", zaxis_title="Z (fwd)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=0, y=-1.5, z=-2.5),
                up=dict(x=0, y=-1, z=0),
                center=dict(x=cx, y=cy, z=cz),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0), height=420,
    )
    return fig


def _render_mesh_plotly(data: Dict, frame_idx: int) -> go.Figure:
    if data.get("meshes") is None:
        return go.Figure().update_layout(title="no mesh data", paper_bgcolor="#111")
    image_ids = data["image_ids"]
    if not (0 <= frame_idx < len(image_ids)):
        return go.Figure().update_layout(title="frame out of range", paper_bgcolor="#111")

    image_id = image_ids[frame_idx]
    mesh_ids = data["meshes"]["image_ids"]
    match = np.where(mesh_ids == image_id)[0]
    if match.size == 0:
        return go.Figure().update_layout(
            title=f"image_id {image_id} not in mesh NPZ", paper_bgcolor="#111")

    verts = data["meshes"]["vertices"][int(match[0])]
    cam_t = np.asarray(data["by_id"][image_id].get("pred_cam_t", [0, 0, 0]))
    verts = verts + cam_t[None, :]
    faces = data["meshes"].get("faces")

    fig = go.Figure()
    if faces is not None:
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="lightblue", opacity=1.0, flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.1),
            hoverinfo="skip",
        ))
    else:
        subset = verts[::8]
        fig.add_trace(go.Scatter3d(
            x=subset[:, 0], y=subset[:, 1], z=subset[:, 2],
            mode="markers", marker=dict(size=1.5, color="lightblue", opacity=0.6),
            hoverinfo="skip", showlegend=False,
        ))

    cx, cy, cz = float(verts[:, 0].mean()), float(verts[:, 1].mean()), float(verts[:, 2].mean())
    fig.update_layout(
        title=f"mesh — frame {frame_idx + 1}/{len(image_ids)} (drag to rotate)",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data", bgcolor="#111",
            camera=dict(
                eye=dict(x=cx, y=cy, z=cz - 2.5),
                up=dict(x=0, y=-1, z=0),
                center=dict(x=cx, y=cy, z=cz),
                projection=dict(type="orthographic"),
            ),
        ),
        paper_bgcolor="#111", font=dict(color="#ccc"),
        margin=dict(l=0, r=0, t=40, b=0), height=420,
    )
    return fig


def _render_timeseries(data: Dict, frame_idx: int) -> go.Figure:
    sigs = data.get("signals", {})
    fig = go.Figure()
    for name, arr in sigs.items():
        if name not in _SIGNAL_COLORS:
            continue
        fig.add_trace(go.Scatter(
            x=list(range(len(arr))), y=arr.tolist(),
            name=name, mode="lines",
            line=dict(color=_SIGNAL_COLORS[name], width=2),
        ))
    fig.add_vline(x=frame_idx, line=dict(color="white", dash="dash", width=1))
    fig.update_layout(
        title=f"Angle signals — frame {frame_idx} marked",
        xaxis_title="frame", yaxis_title="degrees",
        height=280, margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", y=-0.25),
        paper_bgcolor="#1a1a1a", plot_bgcolor="#1a1a1a",
        font=dict(color="#ccc"),
    )
    return fig


def _render_angles_md(data: Dict, frame_idx: int) -> str:
    sigs = data.get("signals", {})
    if not sigs:
        return "*no signal data*"
    rows = []
    for name, arr in sigs.items():
        if frame_idx >= len(arr):
            continue
        v = arr[frame_idx]
        val = "—" if np.isnan(v) else f"{v:+.1f}°"
        rows.append(f"| {name} | {val} |")
    return (
        f"**Frame {frame_idx} angles:**\n\n"
        "| signal | value |\n|---|---|\n"
        + "\n".join(rows)
    )


def _availability_md(video_id: str, rep_index: int) -> str:
    sam3d_dir = _sam3d_output_dir(video_id, rep_index)
    if sam3d_dir:
        return f"✅ SAM3D: `{sam3d_dir}`"
    searched = _sam3d_searched_dirs(video_id, rep_index)
    return "❌ No SAM3D data. Searched:\n" + "\n".join(f"- `{d}`" for d in searched)


# ---------------------------------------------------------------------------
# Build app
# ---------------------------------------------------------------------------

def make_app() -> gr.Blocks:
    with gr.Blocks(title="3D Mesh Viewer", theme=gr.themes.Base()) as app:
        gr.Markdown(
            "## 3D Mesh Viewer\n"
            "Overlay video player · Interactive 3D skeleton & mesh · Per-frame angle signals"
        )

        # ── Session state ──
        pose3d_state = gr.State(None)  # holds _load_pose3d_extras result

        # ── Controls row ──
        with gr.Row():
            video_dd = gr.Dropdown(
                label="Video", choices=_VIDEO_CHOICES,
                value=_VIDEO_CHOICES[0][1] if _VIDEO_CHOICES else None,
                interactive=True, scale=3,
            )
            rep_dd = gr.Dropdown(
                label="Rep",
                choices=_rep_choices(_VIDEO_CHOICES[0][1]) if _VIDEO_CHOICES else [],
                value="0", interactive=True, scale=1,
            )
            mode_radio = gr.Radio(
                label="Overlay mode", choices=_MODE_CHOICES,
                value="raw", interactive=True, scale=2,
            )
            render_btn = gr.Button("▶ Play / render", variant="primary", scale=1)

        avail_md = gr.Markdown("*Select a video.*")

        # ── Top row: video player + frame slider ──
        with gr.Row():
            with gr.Column(scale=2):
                video_player = gr.Video(label="Overlay video", height=360)
                stats_md = gr.Markdown("")
            with gr.Column(scale=1):
                frame_slider = gr.Slider(
                    label="Frame (for 3D panels below)",
                    minimum=0, maximum=1, step=1, value=0,
                    interactive=True,
                )
                angles_md = gr.Markdown("*select a frame*")

        # ── Middle row: interactive 3D panels ──
        with gr.Row():
            skel_plot = gr.Plot(label="3D skeleton (drag to rotate)")
            mesh_plot = gr.Plot(label="Fitted mesh (drag to rotate)")

        # ── Bottom: time-series ──
        ts_plot = gr.Plot(label="Angle signals across rep")

        # ── Gallery (raw frames, always raw) ──
        gallery = gr.Gallery(label="Raw frames", columns=8, height=140,
                             object_fit="contain")

        # ────────────────────────────────────────────────────────────────────
        # Event handlers
        # ────────────────────────────────────────────────────────────────────

        def on_video_change(video_id):
            choices = _rep_choices(video_id)
            val = choices[0] if choices else "0"
            return gr.update(choices=choices, value=val)

        def _load_state(video_id, rep_str):
            """(Re)load pose3d data for (video, rep) and return (state, avail_md, frame_max)."""
            if not video_id or not rep_str:
                return None, "*no video*", gr.update(maximum=1, value=0)
            rep_index = int(rep_str)
            data = _load_pose3d_extras(video_id, rep_index)
            avail = _availability_md(video_id, rep_index)
            n = len(data["image_ids"]) if data else 0
            return data, avail, gr.update(maximum=max(1, n - 1), value=0)

        def _render_video(video_id, rep_str, mode):
            if not video_id or not rep_str:
                return None, ""
            rep_index = int(rep_str)
            path = _resolve_video_for_mode(video_id, rep_index, mode)
            stats = ""
            if mode in _3D_MODES:
                s = _load_mesh_render_stats(video_id, rep_index, mode)
                if s:
                    stats = RenderStats(**s).summary_md()
            return path, stats

        def _render_3d_panels(data, frame_idx):
            if data is None:
                empty = go.Figure()
                return empty, empty, empty, "*no data*"
            fi = int(frame_idx)
            skel = _render_3d_skeleton_plotly(data, fi)
            mesh = _render_mesh_plotly(data, fi)
            ts = _render_timeseries(data, fi)
            angles = _render_angles_md(data, fi)
            return skel, mesh, ts, angles

        # Render button: load video + state + 3D panels
        def on_render(video_id, rep_str, mode):
            data, avail, slider_upd = _load_state(video_id, rep_str)
            video_path, stats = _render_video(video_id, rep_str, mode)
            skel, mesh, ts, angles = _render_3d_panels(data, 0)
            gallery_frames = get_gallery_frames(
                video_id, rep_index=int(rep_str) if rep_str else 0,
                max_frames=24, skeleton=False) if video_id else []
            return (video_path, stats, data, avail, slider_upd,
                    skel, mesh, ts, angles, gallery_frames)

        render_btn.click(
            fn=on_render,
            inputs=[video_dd, rep_dd, mode_radio],
            outputs=[video_player, stats_md, pose3d_state, avail_md,
                     frame_slider, skel_plot, mesh_plot, ts_plot, angles_md, gallery],
        )

        _RENDER_OUTPUTS = [video_player, stats_md, pose3d_state, avail_md,
                           frame_slider, skel_plot, mesh_plot, ts_plot, angles_md, gallery]

        # Video change: (1) refresh the rep list + reset to its first rep, THEN
        # (2) fully re-render so the panels/video reflect the NEW video. The .then()
        # chain guarantees the re-render even when the new rep value equals the old
        # one (a plain rep_dd.change wouldn't fire on an unchanged value).
        video_dd.change(
            on_video_change, inputs=video_dd, outputs=rep_dd,
        ).then(
            fn=on_render,
            inputs=[video_dd, rep_dd, mode_radio],
            outputs=_RENDER_OUTPUTS,
        )

        # Rep/mode change: re-render
        for trigger in [rep_dd, mode_radio]:
            trigger.change(
                fn=on_render,
                inputs=[video_dd, rep_dd, mode_radio],
                outputs=_RENDER_OUTPUTS,
            )

        # Frame slider: update 3D panels only (no re-render of video)
        def on_frame_change(data, frame_idx):
            return _render_3d_panels(data, int(frame_idx))

        frame_slider.change(
            fn=on_frame_change,
            inputs=[pose3d_state, frame_slider],
            outputs=[skel_plot, mesh_plot, ts_plot, angles_md],
        )

        # Startup: load first video
        def _startup():
            if not _VIDEO_CHOICES:
                return (None, "", None, "*no videos*",
                        gr.update(maximum=1, value=0),
                        go.Figure(), go.Figure(), go.Figure(), "*no data*", [])
            return on_render(_VIDEO_CHOICES[0][1], "0", "raw")

        app.load(
            fn=_startup,
            outputs=[video_player, stats_md, pose3d_state, avail_md,
                     frame_slider, skel_plot, mesh_plot, ts_plot, angles_md, gallery],
        )

    return app


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=7871)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    # Gradio only serves files under allowed_paths. Include the static roots plus
    # every registered _KNOWN_VIDEOS data/extra root (e.g. the 2504_processed tree)
    # so raw frames + rendered overlays resolve for any wired session.
    _allowed = [
        "/mnt/data/shared/vlm/data/10k/all",
        "/mnt/data/sgsilva/tmp",
        "/mnt/data/sgsilva/results/sam3dbody_audit",
        CONFIG["video_cache_dir"],
    ]
    _allowed += list(CONFIG.get("extra_data_roots") or [])
    _allowed += list((CONFIG.get("sam3d_video_roots") or {}).values())
    make_app().launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        allowed_paths=sorted(set(_allowed)),
    )
