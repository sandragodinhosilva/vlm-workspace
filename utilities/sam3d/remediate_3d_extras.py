#!/usr/bin/env python3
"""Given a bad-list TSV from audit_3d_extras.py, back up the BAD extras files,
delete them, and emit a remediation manifest the runner can consume.

Reads a TSV with columns: session, rep, n_rogue, kind
  kind ∈ {misfile, no_src_rep, empty, parse_err, missing}

For non-missing kinds: file exists and is bad — copy to backup, then unlink.
For missing: no file to delete; just add to the remediation manifest so the
runner produces it.

The remediation manifest lands at the path passed via --manifest-out and is in
the runner's expected shape:
    {"pairs": [[sess, rep, "processed"], ...],
     "frames_root_tag": "processed",
     "include_rep0": True,
     "rebuild_reason": "<--reason text>"}

Example:
    /home/sgsilva/vlm-post-training-home-venv/bin/python \\
      ~/utilities/sam3d/remediate_3d_extras.py \\
      --data-root /mnt/data/shared/vlm/data/human_annotations/1806_after_format_review_processed \\
      --bad-list /mnt/data/sgsilva/tmp/1806_after_format_review_processed_badlist.tsv \\
      --backup-dir /mnt/data/sgsilva/tmp/1806_bad_backup_2026-06-23 \\
      --manifest-out /mnt/data/sgsilva/tmp/vo3d/manifest_1806_remediation.json \\
      --reason "1806 post-batch remediation"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def remediate(data_root: Path, bad_list: Path, backup_dir: Path,
              manifest_out: Path, reason: str) -> int:
    backup_dir.mkdir(parents=True, exist_ok=True)
    with bad_list.open() as f:
        next(f)  # header
        rows = [l.strip().split("\t") for l in f if l.strip()]

    deleted = backed_up = missing_already = manifest_pairs = 0
    pairs: list[list] = []
    for row in rows:
        if len(row) < 4:
            continue
        sess, rep_s, _n_rogue, kind = row[0], row[1], row[2], row[3]
        rep = int(rep_s)
        path = (data_root / sess / "cropped_repetitions_3d"
                / f"repetition_{rep}" / f"repetition_{rep}_vitpose_3d_extras.json")
        if kind == "missing":
            # No file to delete; just queue for re-run.
            if not path.is_file():
                missing_already += 1
            pairs.append([sess, rep, "processed"])
            manifest_pairs += 1
            continue
        # All other BAD kinds: back up + delete + queue.
        if path.is_file():
            bak = backup_dir / sess / f"repetition_{rep}"
            bak.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bak / path.name)
            path.unlink()
            backed_up += 1
            deleted += 1
        # We still queue it regardless — the source rep dir may exist
        # (misfile/empty/parse_err) and need a fresh annotation. For no_src_rep
        # the runner's _rep_frames_dir will return None and report "no frames",
        # which is the correct outcome (data really is missing upstream).
        pairs.append([sess, rep, "processed"])
        manifest_pairs += 1

    manifest = {
        "pairs": pairs,
        "frames_root_tag": "processed",
        "include_rep0": True,
        "rebuild_reason": reason,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2))

    print(f"bad-list rows:           {len(rows)}")
    print(f"backed up + deleted:     {backed_up}")
    print(f"queued in manifest:      {manifest_pairs}")
    print(f"missing-already (no-op): {missing_already}")
    print(f"backup dir:              {backup_dir}")
    print(f"manifest:                {manifest_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--bad-list", type=Path, required=True)
    ap.add_argument("--backup-dir", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--reason", type=str, default="3D extras remediation")
    args = ap.parse_args()
    return remediate(args.data_root, args.bad_list, args.backup_dir,
                     args.manifest_out, args.reason)


if __name__ == "__main__":
    sys.exit(main())
