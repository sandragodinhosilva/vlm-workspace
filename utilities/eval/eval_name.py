#!/usr/bin/env python3
"""eval_name.py — THE canonical builder/validator for VO eval result filenames.

Stabilization step 3 (2026-07-10, plan:
~/.claude/reports/infra_tooling/2026-07-10_eval_pipeline_stabilization_proposal.md).
Grammar registered in /nomenclature. Hand-templated stems in campaign scripts are what
caused the flipfix-840 split row, the doubled `_thinkon_thinkon` files, and the
dash/underscore invisibility class — build names HERE, never by hand.

USAGE
  # build — print the canonical filename for an artifact:
  eval_name.py build --ckpt /mnt/data/sgsilva/models/qwen35-27b-...-step840 \
      --axis stage2 --thinking on --cohort 1806 --arm gtobsbuild
  # check — validate an existing/planned filename against the grammar (exit 1 on violation):
  eval_name.py check stage2_..._1806_gtobsbuild_thinkon.json [more.json ...]

Conventions reproduced EXACTLY from the current writers (do not "improve" them —
the board compiler's fallback parses these):
  singlestage : {base}[_{cohort}]_singlestage_think{T}.json          (eval_all.sh)
  obs         : obs_{base}[_{cohort}]_think{T}.json                  (eval_all.sh)
  agreement   : agreement_{base}[_{cohort}]_think{T}.json            (eval_all.sh)
  stage2      : stage2_{base}[_{cohort}][_{arm}[_{arm_detail}]]_think{T}.json
                                                                     (reasoner_sweep / campaign)
{base} = FULL basename of the served checkpoint — never strip the qwen35-<N>b- prefix
(stripping it caused the 2026-06-22 4B/27B stem-collision data loss).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compile_eval_results as cer  # the routing truth — namer and router share it

AXES = ("singlestage", "obs", "agreement", "stage2")
ARMS = ("gtobsbuild", "modelobs", "selfloop")
COHORTS = ("1105", "1806")
_THINK_RE = re.compile(r"_think(on|off)")


def build(ckpt: str, axis: str, thinking: str, cohort: str = "",
          arm: str = "", arm_detail: str = "") -> str:
    base = Path(ckpt.rstrip("/")).name
    if base == "hf":
        raise SystemExit("[eval_name] FAIL: basename is 'hf' (a pmartins .../step_N/hf path) — "
                         "pass the SERVED_ID / _ext/<run_id> symlink instead; 'hf' collides "
                         "every such checkpoint's outputs.")
    if axis not in AXES:
        raise SystemExit(f"[eval_name] FAIL: axis {axis!r} not in {AXES}")
    if thinking not in ("on", "off"):
        raise SystemExit(f"[eval_name] FAIL: thinking {thinking!r} must be on|off")
    if cohort and cohort not in COHORTS:
        raise SystemExit(f"[eval_name] FAIL: cohort {cohort!r} not wired (only {COHORTS}); "
                         "wire the new cohort end-to-end first (GT source, floors, testsets)")
    if arm and arm not in ARMS:
        raise SystemExit(f"[eval_name] FAIL: arm {arm!r} not in {ARMS} — a NEW arm needs "
                         "resolve_vo()/EVAL_MAP wiring first, not just a filename")
    if arm and axis != "stage2":
        raise SystemExit("[eval_name] FAIL: --arm only applies to --axis stage2")
    if _THINK_RE.search(base.lower()):
        raise SystemExit(f"[eval_name] FAIL: checkpoint basename {base!r} already contains a "
                         "think token — this is how doubled `_thinkon_thinkon` files happen")
    stem = base + (f"_{cohort}" if cohort else "")
    if axis == "singlestage":
        name = f"{stem}_singlestage_think{thinking}.json"
    elif axis == "obs":
        name = f"obs_{stem}_think{thinking}.json"
    elif axis == "agreement":
        name = f"agreement_{stem}_think{thinking}.json"
    else:  # stage2
        armpart = (f"_{arm}" + (f"_{arm_detail}" if arm_detail else "")) if arm else ""
        name = f"stage2_{stem}{armpart}_think{thinking}.json"
    problems = check(name)
    if problems:  # the builder must never emit a name its own checker rejects
        raise SystemExit("[eval_name] INTERNAL: built name fails check: " + "; ".join(problems))
    return name


def check(filename: str) -> list[str]:
    """Return a list of grammar violations ('' = clean). Lexical + routing-consistency:
    everything knowable WITHOUT the file existing."""
    name = Path(filename).name
    low = name.lower()
    problems = []
    if not low.endswith(".json"):
        problems.append("must end in .json")
    thinks = _THINK_RE.findall(low)
    if len(thinks) == 0:
        problems.append("missing _thinkon/_thinkoff tag (same ckpt gives different numbers "
                        "per thinking mode — feedback_eval_gotchas §3)")
    elif len(thinks) > 1:
        problems.append(f"{len(thinks)} think tags (doubled — e.g. the on-disk "
                        "stage2_sft_step2812_vo_thinkon_thinkon.json mistake)")
    rt = cer.resolve_vo(low)
    # obs_ files are stage-1 artifacts: valid outputs, deliberately NOT scored board rows —
    # resolve_vo() returns kind=None for them, so accept the prefix lexically here.
    if rt["kind"] is None and not low.startswith("obs_"):
        problems.append("no axis marker (stage2_/agreement_/obs_ prefix or singlestage) — "
                        "the board compiler will never read this file")
    # cohort buried where the compiler's anchored regex can't see it:
    raw_cohorts = set(re.findall(r"_(1105|1806)_", low))
    if raw_cohorts and not rt["cohort"] and rt["kind"] in ("two_stage", "single_stage", "agreement"):
        # a cohort-looking tag exists but vo_cohort() didn't recognize it — either it's a
        # training-data tag (fine, but then the file is cohort-untagged: is that intended?)
        # or the eval-cohort tag is in the wrong position (the flipfix-840 class).
        problems.append(f"contains {sorted(raw_cohorts)} but the compiler recognizes NO cohort "
                        "— if this IS a cohort-scoped eval, the tag must sit immediately before "
                        "_singlestage/_think/_gtobsbuild/_modelobs/_selfloop")
    # unknown 4-digit cohort-like tag in a recognized position would route to '' silently:
    m4 = re.search(r"_(\d{4})(?=_think|_gtobsbuild|_gtobs_|_modelobs|_selfloop|_singlestage|\.json$)", low)
    if m4 and m4.group(1) not in COHORTS:
        problems.append(f"cohort-position tag {m4.group(1)!r} is not a wired cohort {COHORTS}")
    for arm in ARMS:
        if arm in low and rt["kind"] == "two_stage" and not rt["cohort"]:
            problems.append(f"arm token {arm!r} without a recognized cohort tag — the compiler's "
                            "arm routing is cohort-gated; this would land on the plain row")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="print the canonical filename")
    b.add_argument("--ckpt", required=True, help="served checkpoint path (SERVED_ID) or basename")
    b.add_argument("--axis", required=True, choices=AXES)
    b.add_argument("--thinking", required=True, choices=("on", "off"))
    b.add_argument("--cohort", default="", help="1105|1806 (omit for legacy single-cohort)")
    b.add_argument("--arm", default="", help="stage2 only: gtobsbuild|modelobs|selfloop")
    b.add_argument("--arm-detail", default="", help="optional arm qualifier (e.g. the obs-model tag)")
    c = sub.add_parser("check", help="validate filename(s); exit 1 on any violation")
    c.add_argument("files", nargs="+")
    args = ap.parse_args()
    if args.cmd == "build":
        print(build(args.ckpt, args.axis, args.thinking, args.cohort, args.arm, args.arm_detail))
        return 0
    bad = 0
    for f in args.files:
        problems = check(f)
        if problems:
            bad += 1
            print(f"✗ {Path(f).name}")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"✓ {Path(f).name}")
    if bad:
        print(f"[eval_name] {bad}/{len(args.files)} filename(s) violate the grammar — fix "
              "BEFORE launching (see /nomenclature 'Eval artifact stem grammar').")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
