#!/usr/bin/env python3
"""Split workflow_tool_use.mmd into per-stage .mmd files that each render landscape.

The full diagram is one `graph TD` with 4 stacked subgraphs -> a ~1:2 strip that is
unreadable on a screen or slide. Each stage on its own, laid out LEFT-TO-RIGHT, fits a
landscape frame.
"""
import pathlib
import re

SRC = pathlib.Path("/home/sgsilva/vlm-post-training/visual_obs/workflow_tool_use.mmd")
OUT = pathlib.Path("/home/sgsilva/tmp/_campaign/20260728_render_mmd/stages")
OUT.mkdir(exist_ok=True)

src = SRC.read_text(encoding="utf-8")

# header = the %%{init}%% theme block (keep verbatim, bump font a touch)
init = src[: src.index("graph TD")].replace("'fontSize':'14px'", "'fontSize':'16px'")
# every classDef / style line, so each fragment keeps the shared palette
styles = "\n".join(
    ln for ln in src.splitlines()
    if ln.strip().startswith(("classDef ", "style "))
)


def block(name: str) -> str:
    """Extract `subgraph NAME[...] ... end` including nested-safe matching."""
    i = src.index(f"subgraph {name}[")
    depth, j = 0, i
    for m in re.finditer(r"^\s*(subgraph\b|end\b)", src[i:], re.M):
        tok = m.group(1)
        depth += 1 if tok == "subgraph" else -1
        if depth == 0:
            j = i + m.end()
            break
    return src[i:j]


def emit(fname: str, title: str, body: str, extra: str = "", direction: str = "LR") -> None:
    """Write a standalone fragment. LR = landscape.

    Only keep `style STAGEn` lines whose subgraph actually appears in this fragment —
    a `style X ...` for an absent id makes mermaid CREATE a phantom empty node.
    """
    keep = []
    for ln in styles.splitlines():
        m = re.match(r"\s*style\s+(\S+)", ln)
        if m and f"subgraph {m.group(1)}[" not in body:
            continue
        keep.append(ln)
    txt = f"{init}graph {direction}\n{body}\n{extra}\n" + "\n".join(keep) + "\n"
    # inside a fragment the stage runs left-to-right too
    txt = txt.replace("direction TB", f"direction {direction}")
    (OUT / fname).write_text(txt, encoding="utf-8")
    print(f"  wrote {fname:34s} ({title})")


# ---- 0 · overview: the 4 stages as one strip, no internals --------------------
overview = """    FL["<b>5 flavours</b><br/>A zero-call · B one-call-many-Q<br/>C spot-wrong-answer · D one-call-one-Q<br/>E several-calls"]:::flavor
    S1["<b>① GENERATION</b><br/>teacher writes the trace<br/>(best of K tries)"]:::gen
    S2["<b>② STAGE-2 REWRITE</b><br/>GT-align the trace<br/><i>reasoning-only: grade is GIVEN,<br/>drift impossible</i>"]:::rewrite
    S3["<b>③ GATE-3 JUDGE</b><br/>3-specialist cascade<br/>J2 format → J1 grounding + J3 purpose"]:::judge
    S4["<b>STAGE-4 REPAIR</b><br/>fix ONLY the flagged defect,<br/>keyed to the failing judge"]:::repair
    K["✅ training set"]:::keep
    D["⛔ set aside<br/>(kept for inspection)"]:::drop
    FL --> S1 --> S2 --> S3
    S3 -->|"all 3 pass"| K
    S3 -->|"any fail"| S4
    S4 -->|"re-judge passes"| K
    S4 -->|"still failing<br/>after budget"| D
    S1 -.->|"shape mismatch"| D
    S2 -.->|"can't ground GT"| D
"""
emit("stage0_overview.mmd", "4-stage overview", overview)

# ---- 1..4 · each stage with its real internals -------------------------------
emit("stage1_generation.mmd", "① generation", block("STAGE1"), extra="""
    FA["<b>A</b>"]:::flavor --> GA
    FB["<b>B</b>"]:::flavor --> GBD
    FD["<b>D</b>"]:::flavor --> GBD
    FE["<b>E</b>"]:::flavor --> GE
    FC["<b>C</b>"]:::flavor --> GC
""")

emit("stage2_rewrite.mmd", "② stage-2 rewrite", block("STAGE2"), extra="""
    IN(["traces from ① generation"]):::gen --> RGATE
    C_OUT(["C trace + wrong cells"]):::gen --> RC
    CONDENSE -->|"no"| OUT3(["→ ③ judge"]):::gate
    RCOND --> OUT3
    RA_SKIP -.-> XD["⛔ set aside"]:::drop
""")

emit("stage3_judge.mmd", "③ gate-3 judge cascade", block("STAGE3"), extra="""
    IN2(["rewritten trace"]):::rewrite --> CHK
    J2 -->|"fail"| TO4(["→ stage-4 repair"]):::repair
    J1J3 -->|"any fail"| TO4
    J1J3 -->|"ALL 3 pass"| KEEP["✅ add to the training set"]:::keep
""")

emit("stage4_repair.mmd", "stage-4 repair", block("STAGE4"), extra="""
    IN3(["failing trace<br/>(from ③ judge)"]):::judge --> S4
    S4F --> RECHK{"re-judge:<br/>all 3 pass?<br/>(within budget)"}:::decision
    S4G --> RECHK
    S4W --> RECHK
    RECHK -->|"yes"| KEEP2["✅ training set"]:::keep
    RECHK -->|"rejected / still<br/>failing after budget"| DROP2["⛔ set aside<br/>(kept for inspection)"]:::drop
""")

print(f"\n{len(list(OUT.glob('*.mmd')))} fragments -> {OUT}")
