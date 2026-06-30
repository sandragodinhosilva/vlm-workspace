"""3D mesh / skeleton overlay renderer for the Video-SFT inspection app.

Provides render_rep_frames() which takes a list of background webp paths and a
SAM-3D-Body output directory, renders each frame in the requested mode, and
returns (png_paths, RenderStats).

Supported modes: "mesh", "3d_skel", "mesh_kp_combined",
                 "side_by_side_raw_mesh", "side_by_side_mesh_skel"

Reuses rasterize_mesh, COCO_SKELETON, COCO_JOINT_SIDE, side_color from
sam3dbody_audit/cropped/render_mesh_and_2d.py (imported at call time to avoid
a hard startup dependency on matplotlib which that module drags in via its
module-level import).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── renderer version — bump this when the rendering algorithm changes so that
#    the cache key changes and stale mp4s are evicted automatically.
RENDERER_VERSION = 1

# ---------------------------------------------------------------------------
# rasterize_mesh — copied from sam3dbody_audit/cropped/render_mesh_and_2d.py
# (that module has a module-level `import matplotlib` which isn't available in
# the app venv; we copy only the pure NumPy+cv2 functions we actually need).
# ---------------------------------------------------------------------------

def _project_verts_mesh(verts_cam: np.ndarray, K: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = verts_cam[:, 0], verts_cam[:, 1], verts_cam[:, 2]
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = K[0, 0] * x / z_safe + K[0, 2]
    v = K[1, 1] * y / z_safe + K[1, 2]
    return np.stack([u, v], axis=-1), z


def rasterize_mesh(verts_cam: np.ndarray, faces: np.ndarray,
                   W: int, H: int, K: np.ndarray,
                   bg_img: Optional[np.ndarray] = None,
                   alpha: float = 0.6) -> np.ndarray:
    """Z-buffered Lambert mesh rasterizer (pure NumPy + cv2)."""
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    if bg_img is not None:
        canvas = bg_img.astype(np.float32).copy()
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    overlay = np.zeros((H, W, 3), dtype=np.float32)

    pts2d, depths = _project_verts_mesh(verts_cam, K)

    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    nrm = np.cross(e1, e2)
    nrm_len = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm_len = np.where(nrm_len < 1e-8, 1.0, nrm_len)
    nrm = nrm / nrm_len
    shading = np.clip(np.abs(nrm[:, 2]) * 0.85 + 0.15, 0.0, 1.0)
    face_depth = (depths[faces[:, 0]] + depths[faces[:, 1]] + depths[faces[:, 2]]) / 3.0
    order = np.argsort(-face_depth)

    base_color = np.array([232, 197, 169], dtype=np.float32)
    p0 = pts2d[faces[:, 0]]
    p1 = pts2d[faces[:, 1]]
    p2 = pts2d[faces[:, 2]]

    for fi in order:
        a, b, c = p0[fi], p1[fi], p2[fi]
        if depths[faces[fi, 0]] <= 0 or depths[faces[fi, 1]] <= 0 or depths[faces[fi, 2]] <= 0:
            continue
        pts = np.stack([a, b, c], axis=0)
        if not np.all(np.isfinite(pts)):
            continue
        x_min = max(0, int(np.floor(pts[:, 0].min())))
        x_max = min(W - 1, int(np.ceil(pts[:, 0].max())))
        y_min = max(0, int(np.floor(pts[:, 1].min())))
        y_max = min(H - 1, int(np.ceil(pts[:, 1].max())))
        if x_max < x_min or y_max < y_min:
            continue
        tri = np.array([[a[0], a[1]], [b[0], b[1]], [c[0], c[1]]], dtype=np.int32)
        local = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        tri_local = tri - np.array([[x_min, y_min]], dtype=np.int32)
        cv2.fillConvexPoly(local, tri_local, 1)
        local_bool = local.astype(bool)
        if not local_bool.any():
            continue
        fd = face_depth[fi]
        zsub = zbuf[y_min:y_max + 1, x_min:x_max + 1]
        update = local_bool & (fd < zsub)
        if not update.any():
            continue
        zsub[update] = fd
        zbuf[y_min:y_max + 1, x_min:x_max + 1] = zsub
        col = base_color * shading[fi]
        ov = overlay[y_min:y_max + 1, x_min:x_max + 1]
        ov[update] = col
        overlay[y_min:y_max + 1, x_min:x_max + 1] = ov
        m = mask[y_min:y_max + 1, x_min:x_max + 1]
        m[update] = True
        mask[y_min:y_max + 1, x_min:x_max + 1] = m

    if bg_img is None:
        out = overlay
    else:
        out = canvas.copy()
        out[mask] = (1.0 - alpha) * canvas[mask] + alpha * overlay[mask]
    return np.clip(out, 0, 255).astype(np.uint8)

# ── MHR-70 bone list (canonical, empirically confirmed on 10073/12002/13003).
#    Single source of truth: keep in sync with audit.py:63-81.
MHR70_BONES: List[Tuple[int, int, str]] = [
    (9, 11, "L"),   # L-hip → L-knee
    (11, 13, "L"),  # L-knee → L-ankle
    (10, 12, "R"),  # R-hip → R-knee
    (12, 14, "R"),  # R-knee → R-ankle
    (9, 10, "C"),   # L-hip ↔ R-hip
    (5, 6, "C"),    # L-shoulder ↔ R-shoulder
    (5, 9, "L"),    # L-shoulder → L-hip
    (6, 10, "R"),   # R-shoulder → R-hip
    (5, 7, "L"),    # L-shoulder → L-elbow
    (6, 8, "R"),    # R-shoulder → R-elbow
    (13, 15, "L"),  # L-ankle → L-foot (indices 15/16/17 all left-side)
    (14, 18, "R"),  # R-ankle → R-foot (indices 18/19/20 all right-side)
]
# Sanity check: if audit.py ever changes its bone list and someone copies it
# here wrong, catch it at import time.
assert MHR70_BONES[10] == (13, 15, "L"), "MHR70_BONES foot entry mismatch"
assert MHR70_BONES[11] == (14, 18, "R"), "MHR70_BONES foot entry mismatch"

LEFT_MHR = {5, 7, 9, 11, 13, 15, 16, 17}
RIGHT_MHR = {6, 8, 10, 12, 14, 18, 19, 20}

# BGR colours for cv2 (OpenCV uses BGR)
SIDE_BGR = {
    "L": (230, 200, 34),   # cyan  #22b1d8 → B,G,R
    "R": (30, 140, 216),   # orange #d8861a → B,G,R
    "C": (40, 220, 240),   # yellow #f0d040 → B,G,R
}

# Off-canvas-joints detection threshold: fraction of KP70 joints that are
# outside the frame before we flag the frame.
_OFF_CANVAS_FRAC_THR = 0.10


@dataclasses.dataclass
class RenderStats:
    n_frames: int = 0
    n_missing_3d: int = 0       # frames where mesh data absent → raw bg used
    n_off_canvas: int = 0       # frames with >10% joints off-canvas
    mode: str = ""

    def summary_md(self) -> str:
        parts = [f"**3D render stats ({self.mode}):** {self.n_frames} frames"]
        if self.n_missing_3d:
            parts.append(f"⚠️ {self.n_missing_3d} missing 3D data")
        if self.n_off_canvas:
            parts.append(f"⚠️ {self.n_off_canvas} off-canvas joints")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Internal: lazy import of rasterize_mesh + COCO helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Internal: load SAM-3D-Body outputs for one rep
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Pose3DData:
    """Parsed SAM-3D-Body outputs for one rep."""
    extras_frames: List[dict]           # list of per-frame dicts from extras JSON
    verts_by_id: Dict[int, np.ndarray]  # image_id → vertices (V,3)
    faces: np.ndarray                   # (T,3) int32
    annotations: Dict[str, dict]        # filename → image record


def _load_pose3d(sam3d_dir: Path) -> Optional[_Pose3DData]:
    """Load extras JSON + meshes NPZ + annotations from a sam3d rep directory.

    Returns None (with a warning) if any required file is missing or corrupt.
    """
    # Find required files by suffix (filename prefix varies by pipeline run)
    def _find(suffix: str) -> Optional[Path]:
        hits = list(sam3d_dir.glob(f"*{suffix}"))
        return hits[0] if hits else None

    extras_path = _find("_3d_extras.json")
    meshes_path = _find("_3d_meshes.npz")
    annot_path = sam3d_dir / "annotations.json"

    if extras_path is None or meshes_path is None:
        return None

    try:
        extras = json.loads(extras_path.read_text())
        npz = np.load(str(meshes_path))
        verts_all = npz["vertices"]          # (F, V, 3)
        faces = npz["faces"].astype(np.int32)
        image_ids = list(npz["image_ids"])

        verts_by_id: Dict[int, np.ndarray] = {}
        for i, img_id in enumerate(image_ids):
            verts_by_id[int(img_id)] = verts_all[i]

        annotations: Dict[str, dict] = {}
        if annot_path.is_file():
            ann_data = json.loads(annot_path.read_text())
            for im in ann_data.get("images", []):
                fname = os.path.basename(im["file_name"])
                annotations[fname] = im

        return _Pose3DData(
            extras_frames=extras.get("frames", []),
            verts_by_id=verts_by_id,
            faces=faces,
            annotations=annotations,
        )
    except Exception as exc:
        print(f"[app_mesh_render] WARNING: failed to load pose3d from {sam3d_dir}: {exc}",
              file=sys.stderr)
        return None


def _build_filename_to_extra(data: _Pose3DData,
                              bg_filenames: Optional[List[str]] = None,
                              ) -> Dict[str, dict]:
    """Map frame filename → extras entry.

    Tries three strategies in order:
    1. annotations.json: filename → image_id → extras entry
    2. Explicit bg_filenames list: positional match (bg_filenames[i] ↔ extras_frames[i])
    3. Nothing found → return empty dict (caller renders raw bg with badge)
    """
    # Strategy 1: via annotations
    id_to_extra = {f["image_id"]: f for f in data.extras_frames}
    result: Dict[str, dict] = {}
    for fname, im in data.annotations.items():
        img_id = im["id"]
        if img_id in id_to_extra:
            result[fname] = id_to_extra[img_id]

    if result:
        return result

    # Strategy 2: positional — caller must pass bg_filenames
    if bg_filenames and data.extras_frames:
        for i, fname in enumerate(bg_filenames):
            if i < len(data.extras_frames):
                result[os.path.basename(fname)] = data.extras_frames[i]
        return result

    return {}


# ---------------------------------------------------------------------------
# Internal: per-frame renderers
# ---------------------------------------------------------------------------

def _project_kp3d(kp3d_cam: np.ndarray, focal: float, W: int, H: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Project (70,3) MHR-70 joints in camera space to pixel coords.

    Returns pts2d (70,2) and z (70,).
    """
    z = kp3d_cam[:, 2]
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = focal * kp3d_cam[:, 0] / z_safe + W / 2.0
    v = focal * kp3d_cam[:, 1] / z_safe + H / 2.0
    return np.stack([u, v], axis=-1), z


def _draw_3d_skeleton_cv(img: np.ndarray, kp3d_cam: np.ndarray,
                         focal: float, W: int, H: int) -> Tuple[np.ndarray, int]:
    """Draw MHR-70 skeleton onto img (in-place copy). Returns (image, n_off_canvas)."""
    out = img.copy()
    pts2d, z = _project_kp3d(kp3d_cam, focal, W, H)

    # Count off-canvas joints (z>0 and either u or v out of bounds)
    valid_z = z > 0
    off_canvas = np.sum(
        valid_z & ((pts2d[:, 0] < 0) | (pts2d[:, 0] >= W) |
                   (pts2d[:, 1] < 0) | (pts2d[:, 1] >= H))
    )
    n_joints = max(1, int(np.sum(valid_z)))
    is_off_canvas_frame = (off_canvas / n_joints) > _OFF_CANVAS_FRAC_THR

    for a, b, side in MHR70_BONES:
        if z[a] <= 0 or z[b] <= 0:
            continue
        pa = pts2d[a]
        pb = pts2d[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        bgr = SIDE_BGR[side]
        cv2.line(out, (int(round(pa[0])), int(round(pa[1]))),
                 (int(round(pb[0])), int(round(pb[1]))),
                 bgr, thickness=2, lineType=cv2.LINE_AA)

    for j in range(min(70, len(pts2d))):
        if z[j] <= 0:
            continue
        p = pts2d[j]
        if not np.isfinite(p).all():
            continue
        if j in LEFT_MHR:
            bgr = SIDE_BGR["L"]
        elif j in RIGHT_MHR:
            bgr = SIDE_BGR["R"]
        else:
            bgr = (180, 180, 180)
        r = 5 if j in {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} else 3
        cv2.circle(out, (int(round(p[0])), int(round(p[1]))), r, bgr, -1, cv2.LINE_AA)

    return out, int(is_off_canvas_frame)


def _draw_2d_skeleton_cv(img: np.ndarray, kp_xy: np.ndarray,
                         scores: np.ndarray, score_thr: float = 0.2) -> np.ndarray:
    """Draw COCO-17 2D skeleton onto img using the canonical bone list."""
    COCO_SKELETON = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [7, 9],
        [6, 8], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4],
    ]
    COCO_SIDE = {0:"C",1:"L",2:"R",3:"L",4:"R",5:"L",6:"R",7:"L",8:"R",
                 9:"L",10:"R",11:"L",12:"R",13:"L",14:"R",15:"L",16:"R"}
    out = img.copy()
    for a, b in COCO_SKELETON:
        if scores[a] < score_thr or scores[b] < score_thr:
            continue
        pa, pb = kp_xy[a], kp_xy[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        sa, sb = COCO_SIDE.get(a, "C"), COCO_SIDE.get(b, "C")
        if sa == sb:
            bgr = SIDE_BGR[sa]
        elif "C" in (sa, sb):
            bgr = SIDE_BGR[sa if sb == "C" else sb]
        else:
            bgr = SIDE_BGR["C"]
        cv2.line(out, (int(round(pa[0])), int(round(pa[1]))),
                 (int(round(pb[0])), int(round(pb[1]))),
                 bgr, thickness=2, lineType=cv2.LINE_AA)
    for j in range(kp_xy.shape[0]):
        if scores[j] < score_thr:
            continue
        p = kp_xy[j]
        if not np.isfinite(p).all():
            continue
        cv2.circle(out, (int(round(p[0])), int(round(p[1]))), 3,
                   SIDE_BGR[COCO_SIDE.get(j, "C")], -1, cv2.LINE_AA)
    return out


def _missing_frame_overlay(bg_bgr: np.ndarray) -> np.ndarray:
    """Return bg with a small 'no 3D data' badge in the top-left corner."""
    out = bg_bgr.copy()
    msg = "no 3D data"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    pad = 4
    cv2.rectangle(out, (pad, pad), (pad + tw + 4, pad + th + 6), (60, 60, 60), -1)
    cv2.putText(out, msg, (pad + 2, pad + th + 2), font, scale,
                (80, 80, 255), thick, cv2.LINE_AA)
    return out


def _off_canvas_badge(img_bgr: np.ndarray) -> np.ndarray:
    """Draw a small red rectangle in the bottom-right corner."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (w - 16, h - 16), (w - 2, h - 2), (0, 0, 220), -1)
    return out


def _make_sentinel_mp4_frames(reason: str, W: int = 640, H: int = 360,
                               n_frames: int = 1) -> List[np.ndarray]:
    """Return a list of BGR frames spelling out a sentinel message."""
    frames = []
    for _ in range(max(1, n_frames)):
        canvas = np.full((H, W, 3), 50, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = reason.split("\n")
        y = H // 2 - len(lines) * 14
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, 0.55, 1)
            x = max(8, (W - tw) // 2)
            cv2.putText(canvas, line, (x, y), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            y += 28
        frames.append(canvas)
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_rep_frames(
    bg_image_paths: List[str],
    sam3d_dir: Path,
    mode: str,
    out_dir: Path,
    coco2d_paths: Optional[List[str]] = None,
) -> Tuple[List[str], RenderStats]:
    """Render a rep's frames in the requested 3D mode.

    Args:
        bg_image_paths: Ordered list of background webp paths (from _rep_filenames).
        sam3d_dir:       SAM-3D-Body output directory for this rep.
        mode:            One of "mesh", "3d_skel", "mesh_kp_combined",
                         "side_by_side_raw_mesh", "side_by_side_mesh_skel".
        out_dir:         Directory to write rendered PNG frames into.
        coco2d_paths:    Optional per-frame COCO-17 JSON paths (for mesh_kp_combined).
                         If None, 2D overlay is skipped for combined modes.

    Returns:
        (png_paths, stats)
    """
    stats = RenderStats(n_frames=len(bg_image_paths), mode=mode)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_pose3d(sam3d_dir)
    fn_to_extra: Dict[str, dict] = {}
    if data is not None:
        fn_to_extra = _build_filename_to_extra(data, bg_filenames=bg_image_paths)

    # (rasterize_mesh is defined locally in this module — no external import needed)

    png_paths: List[str] = []

    for i, bg_path in enumerate(bg_image_paths):
        fname = os.path.basename(bg_path)
        out_path = str(out_dir / f"frame_{i:05d}.png")

        bg_bgr = cv2.imread(bg_path)
        if bg_bgr is None:
            # Unreadable source frame — write a grey placeholder
            bg_bgr = np.full((360, 640, 3), 50, dtype=np.uint8)

        H, W = bg_bgr.shape[:2]
        bg_rgb = bg_bgr[:, :, ::-1]

        # Look up the per-frame 3D data
        extra = fn_to_extra.get(fname)
        has_3d = extra is not None and data is not None

        if not has_3d:
            stats.n_missing_3d += 1
            rendered = _missing_frame_overlay(bg_bgr)
            cv2.imwrite(out_path, rendered)
            png_paths.append(out_path)
            continue

        cam_t = np.asarray(extra["pred_cam_t"], dtype=np.float32)
        focal = float(extra["focal_length"])
        kp3d = np.asarray(extra["pred_keypoints_3d"], dtype=np.float32)  # (70,3)
        kp3d_cam = kp3d + cam_t
        img_id = int(extra["image_id"])
        verts = data.verts_by_id.get(img_id)

        K = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1.0]],
                     dtype=np.float32)

        off_canvas_flag = 0

        if mode == "3d_skel":
            rendered_bgr, off_canvas_flag = _draw_3d_skeleton_cv(
                bg_bgr, kp3d_cam, focal, W, H)

        elif mode == "mesh":
            if verts is None:
                stats.n_missing_3d += 1
                rendered_bgr = _missing_frame_overlay(bg_bgr)
            else:
                verts_cam = verts + cam_t
                mesh_rgb = rasterize_mesh(verts_cam, data.faces, W, H, K,
                                             bg_img=bg_rgb, alpha=0.6)
                rendered_bgr = mesh_rgb[:, :, ::-1]

        elif mode == "mesh_kp_combined":
            if verts is None:
                stats.n_missing_3d += 1
                rendered_bgr = _missing_frame_overlay(bg_bgr)
            else:
                verts_cam = verts + cam_t
                mesh_rgb = rasterize_mesh(verts_cam, data.faces, W, H, K,
                                             bg_img=bg_rgb, alpha=0.6)
                mesh_bgr = mesh_rgb[:, :, ::-1]
                # Overlay 3D skeleton on top of mesh
                rendered_bgr, off_canvas_flag = _draw_3d_skeleton_cv(
                    mesh_bgr, kp3d_cam, focal, W, H)

        elif mode == "side_by_side_raw_mesh":
            if verts is None:
                stats.n_missing_3d += 1
                rendered_bgr = _missing_frame_overlay(bg_bgr)
            else:
                verts_cam = verts + cam_t
                mesh_rgb = rasterize_mesh(verts_cam, data.faces, W, H, K,
                                             bg_img=bg_rgb, alpha=0.6)
                mesh_bgr = mesh_rgb[:, :, ::-1]
                rendered_bgr = np.hstack([bg_bgr, mesh_bgr])

        elif mode == "side_by_side_mesh_skel":
            if verts is None:
                stats.n_missing_3d += 1
                rendered_bgr = _missing_frame_overlay(bg_bgr)
            else:
                verts_cam = verts + cam_t
                mesh_rgb = rasterize_mesh(verts_cam, data.faces, W, H, K,
                                             bg_img=bg_rgb, alpha=0.6)
                mesh_bgr = mesh_rgb[:, :, ::-1]
                skel_bgr, off_canvas_flag = _draw_3d_skeleton_cv(
                    bg_bgr, kp3d_cam, focal, W, H)
                rendered_bgr = np.hstack([mesh_bgr, skel_bgr])

        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        if off_canvas_flag:
            stats.n_off_canvas += 1
            rendered_bgr = _off_canvas_badge(rendered_bgr)

        cv2.imwrite(out_path, rendered_bgr)
        png_paths.append(out_path)

    return png_paths, stats


def make_missing_sentinel_video_frames(
    video_id: str,
    rep_index: int,
    mode: str,
    searched_dirs: List[str],
    W: int = 640,
    H: int = 360,
) -> List[np.ndarray]:
    """Return a list of BGR frames for the 'no SAM3D data' sentinel mp4."""
    lines = [
        "3D output not available.",
        f"video_id: {video_id}  rep: {rep_index}  mode: {mode}",
        "Searched:",
    ] + [f"  {d}" for d in searched_dirs[:4]] + [
        "",
        "Run the SAM-3D-Body pipeline first.",
    ]
    return _make_sentinel_mp4_frames("\n".join(lines), W=W, H=H, n_frames=1)
