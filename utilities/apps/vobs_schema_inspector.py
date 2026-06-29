#!/usr/bin/env python3
"""vobs-schema-inspector — browse a visual-observations schema pair (version-parameterized).

Inspects the two schema files for a chosen --version side-by-side with the PREVIOUS version as the
NEW-vs baseline:
  - visual_observations_<V>_angle.json        (dual-shape: angle Qs as 'angle estimate')
  - visual_observations_<V>_categorical.json  (all-options: angle Qs rewritten to ladders)

Per-exercise cards; NEW-vs-previous-version questions highlighted; the ANGLE vs CATEGORICAL shape of
each angle-type question shown together so you can eyeball the rewrite. Text filter + NEW-only +
angle-only toggles. Pure read-only viewer (no model, no GPU) — the Gradio analogue of the HTML view.

Run via:  launch_app.sh vobs-schema [--version v5_1]   (port 7876; default = latest = v5_1)
"""
import argparse
import json
import os
import html
import re
import collections

REPO = "/home/sgsilva/vlm-post-training"
VO = os.path.join(REPO, "visual_obs")

import gradio as gr

# NEW-vs baseline per version. v2's baseline is the dual v2 file; v3/v4/v5 diff the PREVIOUS
# version's CATEGORICAL file (clean question-text match — avoids the angle-vs-ladder false-NEW).
_PREV = {"v3": "v2", "v4": "v3", "v5": "v4", "v5_1": "v5"}


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _resolve(version):
    """Return (F_ANGLE, F_CAT, F_PREV) for the chosen version."""
    f_angle = os.path.join(VO, f"visual_observations_{version}_angle.json")
    f_cat = os.path.join(VO, f"visual_observations_{version}_categorical.json")
    prev = _PREV.get(version)
    if prev == "v2":
        f_prev = os.path.join(VO, "visual_observations_v2.json")
    elif prev:
        f_prev = os.path.join(VO, f"visual_observations_{prev}_categorical.json")
    else:
        f_prev = None
    return f_angle, f_cat, f_prev


# Module-level state, populated by load_version() at startup (and re-loadable).
ANGLE = CAT = None
PREV_QS = {}       # baseline = immediate previous version
V2_QS = {}         # baseline = v2 (the last version models were trained on) — for the "vs v2" toggle
VERSION = "v5_1"
PREV_LABEL = "v5"
N_EX = N_Q = 0


def _qset_by_ex(schema):
    return {exid: {o["question"] for o in ex.get("vlm_observations", [])}
            for exid, ex in schema.items() if isinstance(ex, dict)}


def load_version(version):
    global ANGLE, CAT, PREV_QS, V2_QS, VERSION, PREV_LABEL, N_EX, N_Q
    f_angle, f_cat, f_prev = _resolve(version)
    ANGLE = _load(f_angle)
    CAT = _load(f_cat)
    prev = _load(f_prev) if f_prev and os.path.exists(f_prev) else {}
    PREV_QS = _qset_by_ex(prev)
    # v2 baseline — the last version used to train models; for the cumulative "vs v2" view.
    # Use the CATEGORICAL v2 file so the 80 angle→ladder rewrites match (categorical-vs-categorical),
    # otherwise every ROM question falsely reads as "new". The card's NEW check uses categorical text.
    f_v2c = os.path.join(VO, "visual_observations_categorical_v2.json")
    f_v2 = os.path.join(VO, "visual_observations_v2.json")
    f_v2_base = f_v2c if os.path.exists(f_v2c) else f_v2
    V2_QS = _qset_by_ex(_load(f_v2_base)) if os.path.exists(f_v2_base) else {}
    VERSION = version
    PREV_LABEL = _PREV.get(version, "—")
    exids = [k for k in ANGLE if k != "_meta"]
    N_EX = len(exids)
    N_Q = sum(len(ANGLE[e].get("vlm_observations", [])) for e in exids)


# back-compat alias used throughout the render code (was V2_QS)
def _prev_qs():
    return PREV_QS


def _opts_html(o):
    if "options" in o:
        chips = "".join(
            f'<span class="chip">{html.escape(str(c))}</span>' for c in o["options"]
        )
        return chips
    return '<span class="chip ang">angle estimate</span>'


def _exercise_card(exid, base_qs, base_label):
    """Render one exercise: angle + categorical shape of each question, NEW-vs-baseline flagged.
    base_qs = per-exercise question-set of the baseline version; base_label = its name for the badge."""
    a = ANGLE[exid]
    c = CAT[exid]
    prevq = base_qs.get(exid, set())
    n_new = 0
    rows = []
    a_obs = a["vlm_observations"]
    c_obs = c["vlm_observations"]
    for i, (oa, oc) in enumerate(zip(a_obs, c_obs)):
        # NEW/MODIFIED = this question is absent from the baseline for this exercise (a reworded
        # question has new text → counts as changed-since-baseline). Compare on the CATEGORICAL text
        # (oc) since the baselines (prev/v2) are categorical — so an angle ROM question matches its
        # v2 ladder and isn't falsely flagged. If no baseline → flag nothing.
        is_new = bool(prevq) and oc["question"] not in prevq
        is_angle = "answer" in oa
        if is_new:
            n_new += 1
        badge = f'<span class="badge">NEW vs {base_label}</span>' if is_new else ""
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
.fam{border:1px solid #c7d2e0;border-radius:12px;padding:10px 12px 4px;margin-bottom:16px;background:#f8fafc}
.fhead{margin-bottom:8px;border-bottom:1px solid #e2e8f0;padding-bottom:6px}
.fname{font-weight:800;color:#0f172a;font-size:15px}
.fmeta{display:block;color:var(--mut);font-size:12px;margin-top:2px}
.fok{color:var(--ac);font-weight:700;margin-left:6px}
.fgap{color:#b91c1c;font-weight:700;margin-left:6px}
"""


def _family(name):
    """Family KEY = exercise name with a trailing (bilateral|left|right) variant suffix stripped,
    lowercased so case-typo'd siblings (e.g. 'Knee Extension' vs 'Knee extension', ex 10081/10082)
    group together. Use _family_label() for display."""
    return re.sub(r"\s*\((bilateral|left|right)\)\s*$", "", (name or "").strip(), flags=re.I).lower()


def _family_label(name):
    """Human-readable family name (case preserved, variant suffix stripped)."""
    return re.sub(r"\s*\((bilateral|left|right)\)\s*$", "", (name or "").strip(), flags=re.I)


def render(query, new_only, angle_only, group_by_family, vs_v2):
    q = (query or "").lower().strip()
    shown = total_new = 0
    # baseline: previous version by default, or v2 (the training baseline) when "vs v2" is ticked.
    base_qs = V2_QS if vs_v2 else PREV_QS
    base_label = "v2" if vs_v2 else PREV_LABEL
    # collect (family, exid, n_new, cardhtml) for everything passing the filters
    items = []
    for exid in sorted(k for k in ANGLE.keys() if k != "_meta"):
        a = ANGLE[exid]
        hay = (exid + " " + a.get("name", "") + " "
               + " ".join(o["question"] for o in a["vlm_observations"])).lower()
        if q and q not in hay:
            continue
        if angle_only and not any("answer" in o for o in a["vlm_observations"]):
            continue
        n_new, cardhtml = _exercise_card(exid, base_qs, base_label)
        if new_only and n_new == 0:
            continue
        items.append((_family(a.get("name", "")), exid, n_new, cardhtml))
        total_new += n_new
        shown += 1

    n_angle = sum(1 for e in ANGLE if e != "_meta"
                  for o in ANGLE[e].get("vlm_observations", []) if "answer" in o)
    # total distinct families across the WHOLE schema (not just the filtered view)
    all_fams = {_family(ANGLE[e].get("name", "")) for e in ANGLE if e != "_meta"}
    n_multi = sum(1 for f in all_fams
                  if sum(1 for e in ANGLE if e != "_meta" and _family(ANGLE[e].get("name", "")) == f) >= 2)

    if group_by_family:
        # group filtered items by family; show a family header with sibling-consistency flag
        fams = {}
        labels = {}
        for fam, exid, n_new, cardhtml in items:
            fams.setdefault(fam, []).append((exid, n_new, cardhtml))
            labels.setdefault(fam, _family_label(ANGLE[exid].get("name", "")))
        blocks = []
        for fam in sorted(fams):
            members = fams[fam]
            ids = [m[0] for m in members]
            # sibling consistency: do all members have the same question count?
            qcounts = {e: len(ANGLE[e].get("vlm_observations", [])) for e in ids}
            consistent = len(set(qcounts.values())) <= 1
            fnew = sum(m[1] for m in members)
            flag = ('<span class="fok">siblings ✓</span>' if consistent
                    else f'<span class="fgap">⚠ sibling gap {qcounts}</span>') if len(ids) > 1 else ""
            fnewb = f'<span class="ncount">+{fnew} new</span>' if fnew else ""
            fhead = (f'<div class="fhead"><span class="fname">{html.escape(labels[fam])}</span> '
                     f'<span class="fmeta">{len(ids)} variant(s): {", ".join(ids)} {flag} {fnewb}</span></div>')
            blocks.append(f'<div class="fam">{fhead}{"".join(m[2] for m in members)}</div>')
        body = "".join(blocks)
        view_note = f' · {len(fams)} families in view'
    else:
        body = "".join(m[3] for m in items)
        view_note = ""

    header = (
        f'<div class="wrap" style="color:var(--mut);margin-bottom:8px">'
        f'{shown} exercises shown · {total_new} questions NEW/changed vs {base_label}'
        f'{" (training baseline)" if vs_v2 else ""} in view{view_note} · '
        f'schema: {N_EX} ex / {N_Q} q ({n_angle} angle) · '
        f'{len(all_fams)} families ({n_multi} multi-sibling)</div>'
    )
    return f'<div class="wrap">{header}{body}</div>'


def _family_qsets():
    """family-key -> (display label, union of question texts across its sibling exercises)."""
    famq, lab = {}, {}
    for k in ANGLE:
        if k == "_meta":
            continue
        f = _family(ANGLE[k].get("name", ""))
        lab.setdefault(f, _family_label(ANGLE[k].get("name", "")))
        famq.setdefault(f, set()).update(_prev_qs_unused(k))
    return lab, famq


def _prev_qs_unused(exid):
    # categorical question texts of an exercise (production canonical)
    return {o["question"] for o in CAT[exid]["vlm_observations"]}


def render_network(threshold, query):
    """Families ↔ shared-questions graph as a vis-network HTML widget.
    Node = family; edge = two families sharing >= threshold question texts (weight = #shared)."""
    import itertools
    threshold = int(threshold)
    lab, famq = _family_qsets()
    fams = sorted(famq)
    q = (query or "").lower().strip()

    edges = []
    for a, b in itertools.combinations(fams, 2):
        sh = len(famq[a] & famq[b])
        if sh >= threshold:
            edges.append((a, b, sh))
    deg = collections.Counter()
    for a, b, s in edges:
        deg[a] += 1
        deg[b] += 1

    # If a text filter is set, keep families matching it + their direct neighbours.
    if q:
        keep = {f for f in fams if q in lab[f].lower()}
        keep |= {b for a, b, s in edges if a in keep} | {a for a, b, s in edges if b in keep}
        edges = [(a, b, s) for a, b, s in edges if a in keep and b in keep]
        nodes_set = keep
    else:
        nodes_set = set(fams)

    # colour by clinical region (rough, from the label) so clusters read at a glance
    def colour(name):
        n = name.lower()
        for kw, c in [("neck", "#f59e0b"), ("shoulder", "#2563eb"), ("wrist", "#0891b2"),
                      ("elbow", "#0ea5e9"), ("hip", "#16a34a"), ("knee", "#22c55e"),
                      ("trunk", "#a855f7"), ("squat", "#84cc16"), ("lunge", "#65a30d"),
                      ("bridge", "#14b8a6"), ("plank", "#ef4444"), ("thoracic", "#8b5cf6"),
                      ("nerve", "#06b6d4")]:
            if kw in n:
                return c
        return "#94a3b8"

    nodes = [{"id": f, "label": lab[f], "value": max(1, deg[f]),
              "color": colour(lab[f]), "title": f"{lab[f]} · degree {deg[f]}"}
             for f in sorted(nodes_set)]
    vis_edges = [{"from": a, "to": b, "value": s, "title": f"{s} shared questions"}
                 for a, b, s in edges]

    n_iso = sum(1 for f in nodes_set if deg[f] == 0)
    data = json.dumps({"nodes": nodes, "edges": vis_edges})
    note = (f'<div class="wrap" style="color:var(--mut);margin:6px 0">'
            f'{len(nodes)} families · {len(vis_edges)} edges (≥{threshold} shared questions) · '
            f'{n_iso} isolated · node size = #connections · colour = body region</div>')
    # self-contained vis-network html (CDN); a fixed-height canvas inside the Gradio HTML slot
    html_doc = f"""{note}
<div id="vobsnet" style="height:680px;border:1px solid #d4d9e0;border-radius:10px;background:#fff"></div>
<script type="text/javascript" src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<script type="text/javascript">
(function(){{
  var d = {data};
  var container = document.getElementById('vobsnet');
  if (!container || typeof vis === 'undefined') return;
  var data = {{nodes: new vis.DataSet(d.nodes), edges: new vis.DataSet(d.edges)}};
  var options = {{
    nodes: {{shape:'dot', scaling:{{min:6,max:34}}, font:{{size:13,face:'-apple-system,Segoe UI'}}}},
    edges: {{color:{{color:'#c7d2e0',highlight:'#2563eb'}}, smooth:false,
             scaling:{{min:0.5,max:6}}, selectionWidth:2}},
    physics: {{stabilization:{{iterations:180}},
               barnesHut:{{gravitationalConstant:-9000,springLength:130,springConstant:0.03,damping:0.5}}}},
    interaction: {{hover:true, tooltipDelay:120, navigationButtons:true, keyboard:false}}
  }};
  new vis.Network(container, data, options);
}})();
</script>"""
    return html_doc


def build_ui():
    with gr.Blocks(title=f"VObs {VERSION} Schema Inspector", css=CSS) as demo:
        gr.Markdown(
            f"## Visual-Observations {VERSION} schema inspector\n"
            f"Browse `visual_observations_{VERSION}_{{angle,categorical}}.json` "
            f"({N_EX} ex / {N_Q} q). Each angle-type question shows "
            f"its **angle** form and its **categorical ladder** together (amber `ANGLE↔CAT` tag); "
            f"questions new/changed vs the baseline get a green `NEW vs <ver>` badge. "
            f"Default baseline = {PREV_LABEL}; tick **vs v2 (training baseline)** to see everything "
            f"added or modified since **v2** — the last version models were trained on."
        )
        with gr.Tab("Browse"):
            with gr.Row():
                query = gr.Textbox(label="Filter (exercise id / name / question text)", scale=4)
                new_only = gr.Checkbox(label="NEW-only", value=False)
                angle_only = gr.Checkbox(label="Angle exercises only", value=False)
                group_by_family = gr.Checkbox(label="Group by family", value=False)
                vs_v2 = gr.Checkbox(label="vs v2 (training baseline)", value=False)
            out = gr.HTML()
            inputs = [query, new_only, angle_only, group_by_family, vs_v2]
            for comp in inputs:
                comp.change(render, inputs=inputs, outputs=out)
            demo.load(render, inputs=inputs, outputs=out)

        with gr.Tab("Family network"):
            gr.Markdown(
                "Families as a **network**: each node is a family (exercise minus L/R/bilateral), "
                "two families are linked when they **share question texts** (edge thickness = how many). "
                "Node size = number of connections; colour = body region. Clusters = clinically-related "
                "movements (e.g. the band/non-band variants, bridge family, wrist family)."
            )
            with gr.Row():
                thr = gr.Slider(2, 8, value=4, step=1, label="Min shared questions to draw an edge")
                netq = gr.Textbox(label="Focus on a family (name filter; shows it + its neighbours)", scale=3)
            net_out = gr.HTML()
            for comp in (thr, netq):
                comp.change(render_network, inputs=[thr, netq], outputs=net_out)
            demo.load(render_network, inputs=[thr, netq], outputs=net_out)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7876)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--version", default="v5_1", choices=["v2", "v3", "v4", "v5", "v5_1"],
                        help="schema version to inspect (default: latest = v5_1)")
    args = parser.parse_args()
    load_version(args.version)
    build_ui().launch(server_name=args.host, server_port=args.port, share=False,
                      theme=gr.themes.Soft())
