#!/usr/bin/env python3
"""Standalone Gradio viewer for llm_prejudge smoke results.

Shows each (video, rep) verdict alongside its frames and the post-hoc label.

Usage:
    cd /home/sgsilva/utilities/apps/video_sft
    source .venv/bin/activate
    python prejudge_viewer.py --port 7870
"""
import argparse
import json
import sys
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Reuse frame helpers from app.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from app import get_gallery_frames, get_or_create_video  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SMOKE_DIR     = Path("/home/sgsilva/tmp")
ABSTRIP_PATH  = Path("/home/sgsilva/tmp/rep_prejudge_smoke_v1_abstrip.jsonl")
POSTHOC_PATH  = Path("/mnt/data/sgsilva/tmp/judge_v14_full.jsonl")

CAT_COLORS = {
    "ok": "#2d9e6b",
    "patient_not_performing": "#d62828",
    "wrong_exercise": "#f77f00",
    "pose_estimation_wrong_subject": "#9b59b6",
    "multi_rep_in_one_rep": "#e67e22",
    "parse_failed": "#888888",
    "error": "#888888",
}


def _available_smoke_files():
    files = sorted(SMOKE_DIR.glob("rep_prejudge_smoke_v*.jsonl"))
    return [str(f) for f in files] or [str(SMOKE_DIR / "rep_prejudge_smoke_v4.jsonl")]


def load_rows(path: Path):
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[(r["video_id"], r["rep_index"])] = r
    return rows


def build_table(pj_rows, gt_rows, cat_filter="All", agree_filter="All"):
    rows = []
    for key, r in sorted(pj_rows.items()):
        gt = gt_rows.get(key, {})
        gt_cat = gt.get("category", "—")
        pj_cat = r.get("category", "?")
        pj_ok = pj_cat == "ok"
        gt_ok = gt_cat == "ok"
        agree = pj_ok == gt_ok
        if cat_filter != "All" and pj_cat != cat_filter:
            continue
        if agree_filter == "Agree" and not agree:
            continue
        if agree_filter == "Disagree" and agree:
            continue
        rows.append({
            "video_id": key[0],
            "rep": key[1],
            "exercise": r.get("exercise_name", "")[:40],
            "pj_category": pj_cat,
            "active_side": r.get("active_side", "?"),
            "side_match": "✓" if r.get("side_match") else "✗",
            "confidence": f"{r.get('confidence', 0):.2f}",
            "gt_category": gt_cat,
            "agree": "✓" if agree else "✗",
        })
    return rows


def format_verdict_md(r, gt, ab_r=None):
    if not r:
        return ""
    pj_cat = r.get("category", "?")
    color = CAT_COLORS.get(pj_cat, "#aaa")
    gt_cat = gt.get("category", "—") if gt else "—"
    gt_color = CAT_COLORS.get(gt_cat, "#aaa")

    md = f"""### `{r['video_id']}` · rep_{r['rep_index']}
**Exercise:** {r.get('exercise_name', '?')}
**Was subsampled:** {r.get('was_subsampled', False)} · **Has skeleton:** {r.get('has_skeleton', '?')}

---
#### Prejudge verdict
**Category:** <span style="color:{color};font-weight:bold">{pj_cat}</span>
**active_side:** `{r.get('active_side','?')}` · expected: `{r.get('expected_side','?')}` · match: {'✓' if r.get('side_match') else '✗'}
**Confidence:** {r.get('confidence', 0):.2f}
**Evidence:** {r.get('evidence', '')}
**Notes:** {r.get('notes') or '—'}
**Prompt version:** {r.get('prejudge_prompt_version', '?')} · **Wall-time:** {r.get('wall_time_s', 0):.2f}s

---
#### Post-hoc judge
**Category:** <span style="color:{gt_color};font-weight:bold">{gt_cat}</span>
**Evidence:** {gt.get('evidence', '') if gt else '—'}
"""
    if ab_r:
        ab_side = ab_r.get("active_side", "?")
        flip = ab_side != r.get("active_side", "?")
        md += f"""
---
#### A/B variant (side-stripped description)
**active_side:** `{ab_side}` {'⚠ FLIPPED' if flip else '(same)'}
**Category:** {ab_r.get('category','?')}
"""
    return md


# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------

def make_app(port: int):
    ab_rows = load_rows(ABSTRIP_PATH)
    gt_rows = load_rows(POSTHOC_PATH)

    smoke_files = _available_smoke_files()
    default_file = next((f for f in smoke_files if "v4" in f), smoke_files[-1])

    cols = ["video_id", "rep", "exercise", "pj_category", "active_side", "side_match", "conf", "gt_category", "agree"]
    all_agrees = ["All", "Agree", "Disagree"]

    def rows_to_table(pj_rows, cat_f, agree_f):
        rows = build_table(pj_rows, gt_rows, cat_f, agree_f)
        tbl = [[
            r["video_id"], r["rep"], r["exercise"],
            r["pj_category"], r["active_side"], r["side_match"],
            r["confidence"], r["gt_category"], r["agree"],
        ] for r in rows]
        total = len(pj_rows)
        shown = len(tbl)
        summary = f"**{shown}** reps shown" + (f" (of {total} total)" if shown != total else "")
        return tbl, summary

    def on_file_change(fpath, cat_f, agree_f):
        pj_rows = load_rows(Path(fpath))
        all_cats = ["All"] + sorted({r["category"] for r in pj_rows.values()})
        n = len(pj_rows)
        fname = Path(fpath).name
        tbl, summary = rows_to_table(pj_rows, cat_f, agree_f)
        return (
            pj_rows,
            gr.update(choices=all_cats, value=cat_f if cat_f in all_cats else "All"),
            tbl,
            summary,
            f"## Prejudge Smoke Viewer · `{fname}` · N={n}\nSorted table of prejudge verdicts. Click a row to inspect frames + evidence.",
            "*Select a row above to inspect.*",
            [],
        )

    def on_filter(pj_rows, cat_f, agree_f):
        tbl, summary = rows_to_table(pj_rows, cat_f, agree_f)
        return tbl, summary

    def on_row_select(evt: gr.SelectData, tbl_data, pj_rows, skel):
        if evt.index is None or len(tbl_data) == 0:
            return "*No row selected.*", []
        row_idx = evt.index[0]
        if row_idx >= len(tbl_data):
            return "*Row out of range.*", []
        import pandas as pd
        if isinstance(tbl_data, pd.DataFrame):
            row = tbl_data.iloc[row_idx].tolist()
        else:
            row = tbl_data[row_idx]
        vid = str(row[0])
        rep = int(row[1])
        key = (vid, rep)
        r = pj_rows.get(key, {})
        gt = gt_rows.get(key, {})
        ab_r = ab_rows.get(key)
        md = format_verdict_md(r, gt, ab_r)
        frames = get_gallery_frames(vid, rep_index=rep, max_frames=24, skeleton=bool(skel))
        return md, frames

    initial_pj = load_rows(Path(default_file))
    initial_cats = ["All"] + sorted({r["category"] for r in initial_pj.values()})
    initial_tbl, initial_summary = rows_to_table(initial_pj, "All", "All")
    initial_n = len(initial_pj)
    initial_fname = Path(default_file).name

    with gr.Blocks(title="Prejudge Smoke Viewer") as app:
        state_pj = gr.State(initial_pj)

        title_md = gr.Markdown(
            f"## Prejudge Smoke Viewer · `{initial_fname}` · N={initial_n}\n"
            "Sorted table of prejudge verdicts. Click a row to inspect frames + evidence."
        )

        with gr.Row():
            file_dd = gr.Dropdown(
                smoke_files, value=default_file, label="Prejudge file", scale=3
            )
            cat_dd = gr.Dropdown(initial_cats, value="All", label="Filter: prejudge category", scale=1)
            agree_dd = gr.Dropdown(all_agrees, value="All", label="Filter: agreement", scale=1)
            use_skel = gr.Checkbox(value=True, label="Skeleton overlay", scale=0)
            refresh_btn = gr.Button("↺ Refresh", scale=0)
            random_btn = gr.Button("⚄ Random", scale=0)

        count_md = gr.Markdown(initial_summary)

        table = gr.Dataframe(
            value=initial_tbl,
            headers=cols,
            interactive=False,
            wrap=False,
        )

        with gr.Row():
            verdict_md = gr.Markdown("*Select a row above to inspect.*")

        with gr.Row():
            gallery = gr.Gallery(label="Rep frames", columns=8, height=200, object_fit="contain")

        def on_refresh(fpath, cat_f, agree_f):
            # Re-read the file from disk (picks up new rows written since startup)
            return on_file_change(fpath, cat_f, agree_f)

        def on_random(pj_rows, cat_f, agree_f, skel):
            import random
            tbl, _ = rows_to_table(pj_rows, cat_f, agree_f)
            if not tbl:
                return "*No rows match the current filter.*", []
            row = random.choice(tbl)
            vid = str(row[0])
            rep = int(row[1])
            key = (vid, rep)
            r = pj_rows.get(key, {})
            gt = gt_rows.get(key, {})
            ab_r = ab_rows.get(key)
            md = format_verdict_md(r, gt, ab_r)
            frames = get_gallery_frames(vid, rep_index=rep, max_frames=24, skeleton=bool(skel))
            return md, frames

        file_dd.change(
            on_file_change,
            [file_dd, cat_dd, agree_dd],
            [state_pj, cat_dd, table, count_md, title_md, verdict_md, gallery],
        )
        cat_dd.change(on_filter, [state_pj, cat_dd, agree_dd], [table, count_md])
        agree_dd.change(on_filter, [state_pj, cat_dd, agree_dd], [table, count_md])
        table.select(on_row_select, [table, state_pj, use_skel], [verdict_md, gallery])
        refresh_btn.click(
            on_refresh,
            [file_dd, cat_dd, agree_dd],
            [state_pj, cat_dd, table, count_md, title_md, verdict_md, gallery],
        )
        random_btn.click(on_random, [state_pj, cat_dd, agree_dd, use_skel], [verdict_md, gallery])

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    make_app(args.port).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Base(),
        allowed_paths=["/mnt/data/shared/vlm/data/10k/all"],
    )
