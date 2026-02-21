#!/usr/bin/env python3
"""
Rename NEST-rPPG/STMap/PURE folders from nest_id (10000, 10003, ...) to dataset3_id
(01-01, 08-03, ...) using pure_nest_to_dataset3_mapping.csv.
Folders with no match or zero overlap are deleted.

Run from repo root or set PURE_ROOT. Use --dry-run to only print actions.
"""
import os
import re
import csv
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))
PURE_ROOT = os.path.join(PROJECT_ROOT, 'NEST-rPPG', 'STMap', 'PURE')
MAPPING_CSV = os.path.join(SCRIPT_DIR, 'pure_nest_to_dataset3_mapping.csv')


def parse_overlap(overlap_info):
    """Parse 'overlap=2333/2333 frames, ...' -> (2333, 2333). Return (0, 0) if not found."""
    m = re.search(r'overlap=(\d+)/(\d+)', overlap_info)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def main():
    ap = argparse.ArgumentParser(description='Rename PURE folders by mapping CSV; delete zero-overlap.')
    ap.add_argument('--dry-run', action='store_true', help='Only print actions, do not rename/delete')
    ap.add_argument('--pure-root', default=PURE_ROOT, help='NEST-rPPG/STMap/PURE root')
    ap.add_argument('--mapping', default=MAPPING_CSV, help='Path to mapping CSV')
    args = ap.parse_args()

    pure_root = os.path.abspath(args.pure_root)
    if not os.path.isdir(pure_root):
        print(f"PURE root not found: {pure_root}")
        return

    if not os.path.isfile(args.mapping):
        print(f"Mapping CSV not found: {args.mapping}")
        return

    # Load mapping and parse overlap
    rows = []
    with open(args.mapping, newline='') as f:
        for row in csv.DictReader(f):
            nest_id = (row.get('nest_id') or '').strip()
            dataset3_id = (row.get('dataset3_id') or '').strip()
            overlap_info = row.get('overlap_info', '')
            overlap_n, _ = parse_overlap(overlap_info)
            rows.append((nest_id, dataset3_id, overlap_n))

    # 1) Folders to delete: no dataset3_id or overlap == 0
    to_delete = set()
    for nest_id, dataset3_id, overlap_n in rows:
        if not nest_id:
            continue
        if not dataset3_id or overlap_n == 0:
            to_delete.add(nest_id)

    # 2) For renames: keep best nest_id per dataset3_id (max overlap)
    ds_to_nest = {}  # dataset3_id -> (nest_id, overlap_n)
    for nest_id, dataset3_id, overlap_n in rows:
        if not nest_id or not dataset3_id or nest_id in to_delete:
            continue
        if dataset3_id not in ds_to_nest or overlap_n > ds_to_nest[dataset3_id][1]:
            ds_to_nest[dataset3_id] = (nest_id, overlap_n)

    renames = [(nest_id, dataset3_id) for dataset3_id, (nest_id, _) in ds_to_nest.items()]

    # 3) Delete zero-overlap / no-match folders
    for nest_id in sorted(to_delete):
        path = os.path.join(pure_root, nest_id)
        if os.path.isdir(path):
            if args.dry_run:
                print(f"[DRY-RUN] Would delete: {path}")
            else:
                import shutil
                shutil.rmtree(path)
                print(f"Deleted: {path}")
        else:
            print(f"Skip delete (not a dir): {path}")

    # 4) Rename nest_id -> dataset3_id (use temp name to avoid conflicts)
    for nest_id, dataset3_id in sorted(renames):
        src = os.path.join(pure_root, nest_id)
        dst = os.path.join(pure_root, dataset3_id)
        if not os.path.isdir(src):
            print(f"Skip rename (source missing): {src}")
            continue
        if os.path.exists(dst) and os.path.realpath(src) != os.path.realpath(dst):
            print(f"Skip rename (target exists): {src} -> {dst}")
            continue
        if nest_id == dataset3_id:
            continue
        if args.dry_run:
            print(f"[DRY-RUN] Would rename: {nest_id} -> {dataset3_id}")
        else:
            os.rename(src, dst)
            print(f"Renamed: {nest_id} -> {dataset3_id}")

    if args.dry_run:
        print("\n[DRY-RUN] No changes made. Run without --dry-run to apply.")


if __name__ == '__main__':
    main()
