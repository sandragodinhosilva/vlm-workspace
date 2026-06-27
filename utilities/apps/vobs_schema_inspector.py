#!/usr/bin/env python3
"""vobs-schema-inspector — browse the v3 visual-observations schema pair.

Inspects the two v3 schema files side-by-side with their v2 baseline:
  - visual_observations_v3_angle.json        (dual-shape: angle Qs as 'angle estimate')
  - visual_observations_v3_categorical.json  (all-options: angle Qs rewritten to ladders)

Per-exercise cards; NEW-vs-v2 questions highlighted; the ANGLE vs CATEGORICAL shape of each
angle-type question shown together so you can eyeball the rewrite. Text filter + NEW-only +
angle-only toggles. Pure read-only viewer (no model, no GPU) — the Gradio analogue of the HTML view.

Run via:  launch_app.sh vobs-schema   (port 7876)
"""
import argparse
import json
import os
import html

REPO = "/home/sgsilva/vlm-post-training"
VO = os.path.join(REPO, "visual_obs")
F_ANGLE = os.path.join(VO, "visual_observations_v3_angle.json")
F_CAT = os.path.join(VO, "visual_observations_v3_categorical.json")
F_V2 = os.path.join(VO, "visual_observations_v2.json")  # baseline to mark NEW v3 questions

import gradio as gr


def _load(path):
    with open(path) as fh:
        return json.load(fh)


ANGLE = _load(F_ANGLE)
CAT = _load(F_CAT)
V2 = _load(F_V2)

# per-exercise set of v2 question texts (to flag NEW-in-v3)
V2_QS = {
    exid: {o["question"] for o in ex.get("vlm_observations", [])}
    for exid, ex in V2.items()
}


def _opts_html(o):
    if "options" in o:
        chips = "".join(
            f'<span class="chip">{html.escape(str(c))}</span>' for c in o["options"]
        )
        return chips
    return '<span class="chip ang">angle estimate</span>'


def _exercise_card(exid):
    """Render one exercise: angle + categorical shape of each question, NEW flagged."""
    a = ANGLE[exid]
    c = CAT[exid]
    v2q = V2_QS.get(exid, set())
    n_new = 0
    rows = []
    a_obs = a["vlm_observations"]
    c_obs = c["vlm_observations"]
    for i, (oa, oc) in enumerate(zip(a_obs, c_obs)):
        is_new = oa["question"] not in v2q
        is_angle = "answer" in oa
        if is_new:
            n_new += 1
        badge = '<span class="badge">NEW v3</span>' if is_new else ""
        atag = '<span class="atag">ANGLE↔CAT</span>' if is_angle else ""
        cls = "q new" if is_new else "q"
        # angle row: show the angle question + its categorical twin if it differs
        if is_angle:
            inner = (
                f'<div class="qt">{html.escape(oa["question"])}{badge}{atag}</div>'
                f'<div class="sub">angle answer: <span class="chip ang">angle estimate</span></div>'
                f'<div class="sub">categorical ladder: <span class="qsub">{html.escape(oc["question"])}</span></div>'
                f'<div class="opts">{_opts_html(oc)}</div>'
            )
        else:
            inner = (
                f'<div class="qt">{html.escape(oa["question"])}{badge}</div>'
                f'<div class="opts">{_opts_html(oa)}</div>'
            )
        rows.append(f'<div class="{cls}" data-new="{int(is_new)}" data-angle="{int(is_angle)}">{inner}</div>')
    cnt = f'<span class="ncount">+{n_new} new</span>' if n_new else ""
    has = "has-new" if n_new else ""
    head = (
        f'<div class="chead"><span class="exid">{exid}</span> '
        f'<span class="exname">{html.escape(a.get("name",""))}</span>'
        f'<span class="meta">{html.escape(a.get("position",""))} · {html.escape(a.get("type",""))} '
        f'· {len(a_obs)} Qs {cnt}</span></div>'
    )
    return n_new, f'<div class="card {has}">{head}{"".join(rows)}</div>'


CSS = """
:root{--card:#ffffff;--new:#eafaf0;--txt:#1a1d24;--mut:#5b6675;--ac:#16a34a;--blu:#2563eb;--amb:#b45309}
.wrap{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--txt)}
.card{background:var(--card);border:1px solid #d4d9e0;border-radius:10px;padding:14px;margin-bottom:12px}
.card.has-new{border-color:#86d9a4;border-left:4px solid var(--ac)}
.chead{margin-bottom:10px}.exid{font-weight:700;color:var(--blu)}.exname{font-weight:600;color:var(--txt)}
.meta{display:block;color:var(--mut);font-size:12px;margin-top:2px}.ncount{color:var(--ac);font-weight:700}
.q{padding:8px 10px;border-radius:7px;background:#f4f6f9;border:1px solid #e5e9ef;margin-bottom:7px}
.q.new{background:var(--new);border-left:3px solid var(--ac)}
.qt{font-weight:600;margin-bottom:5px;color:var(--txt)}.qsub{color:var(--txt)}
.sub{font-size:12px;color:var(--mut);margin:2px 0}
.badge{background:var(--ac);color:#ffffff;font-size:10px;font-weight:800;padding:1px 6px;border-radius:4px;margin-left:7px}
.atag{background:#fde9c8;color:var(--amb);font-size:10px;font-weight:800;padding:1px 6px;border-radius:4px;margin-left:6px}
.opts{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px}
.chip{background:#eef1f6;color:#374151;font-size:12px;padding:2px 9px;border-radius:12px;border:1px solid #dde2ea}
.chip.ang{background:#fdf3e3;color:var(--amb);border-color:#f3dcb8}
"""


def render(query, new_only, angle_only):
    q = (query or "").lower().strip()
    cards = []
    shown = total_new = 0
    for exid in sorted(ANGLE.keys()):
        a = ANGLE[exid]
        # exercise-level text match (id, name, or any question text)
        hay = (exid + " " + a.get("name", "") + " "
               + " ".join(o["question"] for o in a["vlm_observations"])).lower()
        if q and q not in hay:
            continue
        n_new, cardhtml = _exercise_card(exid)
        if new_only and n_new == 0:
            continue
        if angle_only and not any("answer" in o for o in a["vlm_observations"]):
            continue
        total_new += n_new
        shown += 1
        cards.append(cardhtml)
    header = (
        f'<div class="wrap" style="color:var(--mut);margin-bottom:8px">'
        f'{shown} exercises shown · {total_new} NEW-v3 questions in view · '
        f'schema: 241 ex / 1479 q (80 angle)</div>'
    )
    return f'<div class="wrap">{header}{"".join(cards)}</div>'


def build_ui():
    with gr.Blocks(title="VObs v3 Schema Inspector", css=CSS) as demo:
        gr.Markdown(
            "## Visual-Observations v3 schema inspector\n"
            "Browse `visual_observations_v3_{angle,categorical}.json`. Each angle-type question shows "
            "its **angle** form and its **categorical ladder** together (amber `ANGLE↔CAT` tag); "
            "questions new in v3 vs v2 get a green `NEW v3` badge."
        )
        with gr.Row():
            query = gr.Textbox(label="Filter (exercise id / name / question text)", scale=4)
            new_only = gr.Checkbox(label="NEW-only", value=False)
            angle_only = gr.Checkbox(label="Angle exercises only", value=False)
        out = gr.HTML()
        inputs = [query, new_only, angle_only]
        for comp in inputs:
            comp.change(render, inputs=inputs, outputs=out)
        demo.load(render, inputs=inputs, outputs=out)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7876)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    build_ui().launch(server_name=args.host, server_port=args.port, share=False,
                      theme=gr.themes.Soft())
