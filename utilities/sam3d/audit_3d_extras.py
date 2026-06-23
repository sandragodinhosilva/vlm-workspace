#!/usr/bin/env python3
"""Audit SAM-3D extras against their source `cropped_repetitions/repetition_N/`.

For every `*_vitpose_3d_extras.json` under --data-root, check whether the
frame-name set in the extras matches the .webp set in the corresponding source
rep directory. Classifies each extras file as:

  exact      — frames-in-extras == webps-in-src-rep                    (good)
  subset     — frames-in-extras ⊂ webps-in-src-rep                     (good; pipeline drop)
  misfile    — frames-in-extras has names NOT in src-rep dir           (BAD; renumbering / cross-rep)
  no_src_rep — extras dir exists but cropped_repetitions/repetition_N/ is gone (BAD)
  empty      — extras has no frames                                    (BAD)
  parse_err  — JSON parse failed                                       (BAD)

Also reports coverage: every (session, rep) with .webps SHOULD have an extras file.

Outputs:
  stdout summary
  --bad-list-out (default: ./<dataset>_badlist.tsv) — sess\trep\tn_rogue\tkind for everything BAD

This is the canonical SAM-3D audit. It is run BEFORE a batch (to know what's
already-done and trustworthy) AND AFTER a batch (to surface any new misfiles).

Example:
    /home/sgsilva/vlm-post-training-home-venv/bin/python \\
      ~/utilities/sam3d/audit_3d_extras.py \\
      --data-root /mnt/data/shared/vlm/data/human_annotations/1806_after_format_review_processed
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def audit(data_root: Path, bad_list_out: Path | None) -> dict:
    extras = list(data_root.rglob("*_vitpose_3d_extras.json"))
    counts = {"exact": 0, "subset": 0, "misfile": 0,
              "no_src_rep": 0, "empty": 0, "parse_err": 0}
    bad_rows: list[tuple[str, int, int, str]] = []  # (sess, rep, n_rogue, kind)

    for p in extras:
        parts = p.relative_to(data_root).parts
        # parts = (<session>, "cropped_repetitions_3d", "repetition_N", "<file>.json")
        sess = parts[0]
        try:
            rep = int(parts[2].split("_")[-1])
        except (IndexError, ValueError):
            counts["parse_err"] += 1
            bad_rows.append((sess, -1, -1, "parse_err"))
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            counts["parse_err"] += 1
            bad_rows.append((sess, rep, -1, "parse_err"))
            continue
        src_dir = data_root / sess / "cropped_repetitions" / f"repetition_{rep}"
        if not src_dir.is_dir():
            counts["no_src_rep"] += 1
            bad_rows.append((sess, rep, -1, "no_src_rep"))
            continue
        src_names = {wp.name for wp in src_dir.glob("*.webp")}
        ext_names = {
            f["file_name"] for f in d.get("frames", [])
            if isinstance(f, dict) and "file_name" in f
        }
        if not ext_names:
            counts["empty"] += 1
            bad_rows.append((sess, rep, 0, "empty"))
            continue
        if ext_names == src_names:
            counts["exact"] += 1
        elif ext_names.issubset(src_names):
            counts["subset"] += 1
        else:
            n_rogue = len(ext_names - src_names)
            counts["misfile"] += 1
            bad_rows.append((sess, rep, n_rogue, "misfile"))

    # Coverage: every (sess, rep) with .webps must have a matching extras.
    missing: list[tuple[str, int]] = []
    expected = 0
    for sd in sorted(p for p in data_root.iterdir() if p.is_dir()):
        crep = sd / "cropped_repetitions"
        if not crep.is_dir():
            continue
        for rd in crep.iterdir():
            if not rd.is_dir() or not rd.name.startswith("repetition_"):
                continue
            try:
                rep = int(rd.name.split("_")[-1])
            except ValueError:
                continue
            if not any(rd.glob("*.webp")):
                continue
            expected += 1
            ext = (sd / "cropped_repetitions_3d" / rd.name
                   / f"{rd.name}_vitpose_3d_extras.json")
            if not ext.is_file():
                missing.append((sd.name, rep))

    # Print summary.
    n_extras = len(extras)
    print(f"data-root: {data_root}")
    print(f"total extras on disk: {n_extras}")
    print(f"  exact:        {counts['exact']}")
    print(f"  subset-only:  {counts['subset']}  (pipeline drops, normal)")
    print(f"  MISFILE:      {counts['misfile']}")
    print(f"  NO_SRC_REP:   {counts['no_src_rep']}")
    print(f"  empty:        {counts['empty']}")
    print(f"  parse-err:    {counts['parse_err']}")
    print(f"expected pairs (sessions × reps with .webp): {expected}")
    print(f"missing extras (gap to fill): {len(missing)}")

    if bad_rows:
        prefix_count = Counter(r[0].split("_")[0] for r in bad_rows
                               if r[3] in ("misfile", "no_src_rep"))
        if prefix_count:
            print(f"top bad-prefix counts: {prefix_count.most_common(10)}")

    # Persist bad list (misfile + no_src_rep + empty + parse_err) for remediation.
    if bad_list_out is not None and (bad_rows or missing):
        bad_list_out.parent.mkdir(parents=True, exist_ok=True)
        with bad_list_out.open("w") as f:
            f.write("session\trep\tn_rogue\tkind\n")
            for s, r, n, k in bad_rows:
                f.write(f"{s}\t{r}\t{n}\t{k}\n")
            for s, r in missing:
                f.write(f"{s}\t{r}\t0\tmissing\n")
        print(f"wrote bad/missing list -> {bad_list_out}  "
              f"({len(bad_rows)} bad + {len(missing)} missing)")

    ok = (counts["misfile"] == 0
          and counts["no_src_rep"] == 0
          and counts["empty"] == 0
          and counts["parse_err"] == 0
          and len(missing) == 0)
    print()
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    return {"counts": counts, "missing": len(missing),
            "expected": expected, "ok": ok}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True,
                    help="dataset root, e.g. .../1806_after_format_review_processed")
    ap.add_argument("--bad-list-out", type=Path, default=None,
                    help="TSV to write bad/missing pairs (default: "
                         "/mnt/data/sgsilva/tmp/<dataset>_badlist.tsv)")
    args = ap.parse_args()

    if not args.data_root.is_dir():
        print(f"ERROR: --data-root not a directory: {args.data_root}",
              file=sys.stderr)
        return 2
    if args.bad_list_out is None:
        args.bad_list_out = (
            Path("/mnt/data/sgsilva/tmp")
            / f"{args.data_root.name}_badlist.tsv"
        )

    r = audit(args.data_root, args.bad_list_out)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
