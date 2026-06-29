#!/usr/bin/env python
"""Print the total row count of an HF dataset dir for dlog auto-count.

Sums num_rows across splits for a DatasetDict (NOT len(), which returns the split
COUNT); handles a bare Dataset; falls back to summing standard split subdirs
(train/validation/test) when the root has no dataset_dict.json. Exits non-zero on
failure so the caller can emit a distinct "?" sentinel (never a silent 0/happy path).

Usage: count_rows.py <dataset_dir>
"""
import os
import sys

from datasets import DatasetDict, load_from_disk


def total(path: str) -> int:
    d = load_from_disk(path)
    if isinstance(d, DatasetDict):  # DatasetDict.num_rows returns a DICT, not an int
        return sum(v.num_rows for v in d.values())
    return d.num_rows  # bare Dataset


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: count_rows.py <dataset_dir>", file=sys.stderr)
        return 2
    p = sys.argv[1]
    try:
        print(total(p))
        return 0
    except Exception:
        # root not directly loadable: sum standard split subdirs
        parts = []
        for sp in ("train", "validation", "test"):
            sd = os.path.join(p, sp)
            if os.path.isdir(sd):
                try:
                    parts.append(load_from_disk(sd).num_rows)
                except Exception:
                    pass
        if parts:
            print(sum(parts))
            return 0
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
