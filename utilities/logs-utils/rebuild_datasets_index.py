#!/usr/bin/env python3
"""Regenerate ~/.claude/datasets_index.json from ~/.claude/DATASETS.md.

Run after any dlog.sh registration:
    python3 ~/utilities/logs-utils/rebuild_datasets_index.py
"""
import json, re
from datetime import date
from pathlib import Path

DATASETS_MD = Path.home() / '.claude' / 'DATASETS.md'
INDEX_JSON   = Path.home() / '.claude' / 'datasets_index.json'

content = DATASETS_MD.read_text()
raw_blocks = re.split(r'(?m)^(?=### `)', content)

records = []
for block in raw_blocks:
    if not block.strip().startswith('### `'):
        continue
    lines = block.strip().splitlines()
    m = re.search(r'`([^`]+)`\s+\((\d{4}-\d{2}-\d{2})\)', lines[0])
    if not m:
        continue
    name, built_date = m.groups()
    # Default any entry without an explicit Status line to 'canonical' so the
    # field is always present and queryable (legacy entries predate the field).
    rec = {'name': name, 'date': built_date, 'status': 'canonical'}
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
    'source': str(DATASETS_MD),
    'total': len(records),
    'datasets': records,
}

INDEX_JSON.write_text(json.dumps(index, indent=2))
print(f'Wrote {len(records)} datasets to {INDEX_JSON}')
for r in records:
    print(f"  {r['date']}  {r.get('rows', '?'):>8}  {r['name']}")