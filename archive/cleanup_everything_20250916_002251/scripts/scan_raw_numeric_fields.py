#!/usr/bin/env python3
"""
Scan raw coloring JSON files to summarize all numeric fields across the dataset.

Outputs:
- results/raw_numeric/raw_numeric_summary.csv: per-key stats (count, presence_rate, min, mean, std, max)
- results/raw_numeric/raw_numeric_summary.md: human-readable summary

Only processes Coloring_*.json within each child directory.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import math
import numpy as np
import pandas as pd


def is_number(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def scan_raw_numeric(root: Path, limit_children: int | None = None, limit_files_per_child: int | None = None):
    # stats per key
    stats = defaultdict(lambda: {
        'count': 0,
        'sum': 0.0,
        'sumsq': 0.0,
        'min': float('inf'),
        'max': float('-inf'),
        'presence': 0,   # number of points where key was present
    })
    total_points = 0

    children = [d for d in root.iterdir() if d.is_dir()]
    if limit_children:
        children = children[:limit_children]

    for child in children:
        files = sorted(child.glob('Coloring_*.json'))
        if limit_files_per_child:
            files = files[:limit_files_per_child]

        for fp in files:
            try:
                with fp.open('r') as f:
                    data = json.load(f)
                td = data.get('json', {}).get('touchData', {})
                # td is dict of stroke_id -> list of points
                for _, points in td.items():
                    for pt in points or []:
                        total_points += 1
                        for k, v in pt.items():
                            if is_number(v):
                                st = stats[k]
                                st['count'] += 1
                                st['sum'] += float(v)
                                st['sumsq'] += float(v) * float(v)
                                st['min'] = min(st['min'], float(v))
                                st['max'] = max(st['max'], float(v))
                                st['presence'] += 1
            except Exception as e:
                # skip problematic files
                continue

    # build DataFrame
    rows = []
    for k, st in stats.items():
        c = st['count']
        if c == 0:
            continue
        mean = st['sum'] / c
        var = max(0.0, (st['sumsq'] / c) - mean * mean)
        std = math.sqrt(var)
        rows.append({
            'key': k,
            'count': c,
            'presence_rate': st['presence'] / max(1, total_points),
            'min': st['min'],
            'mean': mean,
            'std': std,
            'max': st['max'],
        })

    df = pd.DataFrame(rows).sort_values(['presence_rate', 'count'], ascending=[False, False])
    return df, total_points


def main():
    ap = argparse.ArgumentParser(description='Scan raw coloring JSON numeric fields')
    ap.add_argument('--root', type=str, required=True, help='Root directory containing child folders with Coloring_*.json')
    ap.add_argument('--limit-children', type=int, help='Optional limit on number of child dirs to scan')
    ap.add_argument('--limit-files-per-child', type=int, help='Optional limit on files per child to scan')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path('results') / 'raw_numeric'
    out_dir.mkdir(parents=True, exist_ok=True)

    df, total_points = scan_raw_numeric(root, args.limit_children, args.limit_files_per_child)

    csv_path = out_dir / 'raw_numeric_summary.csv'
    md_path = out_dir / 'raw_numeric_summary.md'

    df.to_csv(csv_path, index=False)

    # simple markdown report
    lines = []
    lines.append('# Raw Numeric Fields Summary')
    lines.append(f'Total touch points scanned: {total_points}')
    lines.append('')
    if not df.empty:
        # Top entries by presence
        top = df.head(20)
        lines.append('## Top 20 most present numeric keys')
        for _, r in top.iterrows():
            lines.append(f"- {r['key']}: presence={r['presence_rate']:.3f}, mean={r['mean']:.4f}, std={r['std']:.4f}, min={r['min']:.4f}, max={r['max']:.4f}")
    else:
        lines.append('No numeric fields found.')

    md_path.write_text('\n'.join(lines))

    print(f'Saved: {csv_path}')
    print(f'Saved: {md_path}')


if __name__ == '__main__':
    main()

