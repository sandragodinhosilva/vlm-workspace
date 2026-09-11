#!/usr/bin/env python3
"""Regenerate ~/.claude/datasets_index.json from the `### entry`-format dataset registries.

Reads VLM_DATASETS.md and REAS_DATASETS.md and merges them, stamping each record with the
registry it came from. 3WC_DATASETS.md is a markdown TABLE, not `### entry` format, so it is
deliberately not parsed here.

2026-09-11: previously read VLM_DATASETS.md alone, so every rebuild silently DELETED the
reasonings datasets that `dlog --registry reas` had written to REAS_DATASETS.md.

Run after any dlog.sh registration:
    python3 ~/utilities/logs-utils/rebuild_datasets_index.py
"""
import json, re
from datetime import date
from pathlib import Path

REGISTRIES = [
    ('vlm',  Path.home() / '.claude' / 'VLM_DATASETS.md'),
    ('reas', Path.home() / '.claude' / 'REAS_DATASETS.md'),
]
INDEX_JSON   = Path.home() / '.claude' / 'datasets_index.json'

raw_blocks = []
for _reg_name, _reg_path in REGISTRIES:
    if not _reg_path.exists():
        print(f'  (registry {_reg_path.name} not found, skipped)')
        continue
    for _b in re.split(r'(?m)^(?=### `)', _reg_path.read_text()):
        raw_blocks.append((_reg_name, _b))

records = []
for _registry, block in raw_blocks:
    if not block.strip().startswith('### `'):
        continue
    lines = block.strip().splitlines()
    m = re.search(r'`([^`]+)`\s+\((\d{4}-\d{2}-\d{2})\)', lines[0])
    if not m:
        continue
    name, built_date = m.groups()
    # Default any entry without an explicit Status line to 'canonical' so the
    # field is always present and queryable (legacy entries predate the field).
    rec = {'name': name, 'date': built_date, 'status': 'canonical', 'registry': _registry}
    for line in lines[1:]:
        for field, key in [('**Status:**', 'status'), ('**Path:**', 'path'),
                           ('**Purpose:**', 'purpose'), ('**Builder:**', 'builder'),
                           ('**Sources:**', 'sources'), ('**Rows:**', 'rows')]:
            if line.strip().startswith(f'- {field}'):
                val = line.strip()[len(f'- {field}'):].strip().strip('`')
                if key == 'status':
                    # "superseded by `X`" → status=superseded, superseded_by=X
                    sm = re.match(r'(\w+)\s*\(superseded by `([^`]+)`\)', val)
                    if sm:
                        rec['status'] = sm.group(1)
                        rec['superseded_by'] = sm.group(2)
                    else:
                        rec['status'] = val.split()[0] if val.split() else 'canonical'
                else:
                    rec[key] = int(val) if key == 'rows' and val.isdigit() else val
    records.append(rec)

records.sort(key=lambda r: r['date'], reverse=True)

index = {
    'generated': str(date.today()),
    'source': ', '.join(p.name for _n, p in REGISTRIES if p.exists()),
    'total': len(records),
    'datasets': records,
}

INDEX_JSON.write_text(json.dumps(index, indent=2))
print(f'Wrote {len(records)} datasets to {INDEX_JSON}')
for r in records:
    print(f"  {r['date']}  {r.get('rows', '?'):>8}  {r['name']}")