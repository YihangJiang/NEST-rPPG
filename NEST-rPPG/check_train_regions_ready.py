# %%
"""
Verify minimal files required by train_regions.py for region domains (*_in, *_rm, *_eye).

Per subject, checks:
  - STMap/STMap_RGB.png exists
  - Label/BVP.mat exists (key 'BVP')
  - Label/HR.mat exists (key 'HR')
  - STMap width == BVP length == HR length (within each region)
  - Those lengths are identical across in / rm / eye

Run cells top-to-bottom in VS Code / Cursor / Jupyter.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import scipy.io as scio

import config

# %%
# --- Config (edit for notebook runs) ---
CHECK_BASES = ('PURE_my', 'UBFC_my', 'BUAA_my')  # or ('BUAA_my',) for one dataset
ONLY_SUBJECT = None  # e.g. "Sub_10lux 15.8"; None = all subjects in all 3 regions
STMAP_NAME = config.STMAP_NAME  # default STMap_RGB.png
VERBOSE = False  # True = print passing subjects too

STMAP_SUBDIR = 'STMap'
LABEL_SUBDIR = 'Label'
REGION_SUFFIXES = ('in', 'rm', 'eye')

# %%
def region_domains(base: str) -> dict[str, str]:
    """PURE_my -> {in: PURE_my_in, rm: PURE_my_rm, eye: PURE_my_eye}."""
    return {suffix: f'{base}_{suffix}' for suffix in REGION_SUFFIXES}


def stmap_my_root(domain: str) -> str:
    if domain not in config.FILEA_NAME:
        raise KeyError(f'Unknown domain {domain!r} (not in config.FILEA_NAME)')
    rel = config.FILEA_NAME[domain][0]
    return os.path.join(config.STMAP_PARENT_ROOT, rel)


def list_subjects(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if not d.startswith('.') and os.path.isdir(os.path.join(root, d))
    )


def _read_stmap_width(stmap_path: str) -> int | None:
    if not os.path.isfile(stmap_path):
        return None
    img = cv2.imread(stmap_path)
    if img is None:
        return None
    return int(img.shape[1])


def _read_mat_len(mat_path: str, key: str) -> int | None:
    if not os.path.isfile(mat_path):
        return None
    try:
        data = scio.loadmat(mat_path)
    except Exception:
        return None
    if key not in data:
        return None
    return int(np.asarray(data[key]).reshape(-1).shape[0])


def check_subject_in_region(subject_root: str, stmap_name: str) -> dict:
    """Return paths, lengths, and issue strings for one subject in one region."""
    stmap_path = os.path.join(subject_root, STMAP_SUBDIR, stmap_name)
    bvp_path = os.path.join(subject_root, LABEL_SUBDIR, 'BVP.mat')
    hr_path = os.path.join(subject_root, LABEL_SUBDIR, 'HR.mat')

    stmap_w = _read_stmap_width(stmap_path)
    bvp_len = _read_mat_len(bvp_path, 'BVP')
    hr_len = _read_mat_len(hr_path, 'HR')

    issues: list[str] = []
    if stmap_w is None:
        issues.append(f'missing or unreadable STMap: {stmap_path}')
    if bvp_len is None:
        issues.append(f'missing or invalid BVP.mat: {bvp_path}')
    if hr_len is None:
        issues.append(f'missing or invalid HR.mat: {hr_path}')

    lengths = [x for x in (stmap_w, bvp_len, hr_len) if x is not None]
    if len(lengths) == 3 and len(set(lengths)) > 1:
        issues.append(f'length mismatch in region: STMap={stmap_w} BVP={bvp_len} HR={hr_len}')

    return {
        'stmap_path': stmap_path,
        'bvp_path': bvp_path,
        'hr_path': hr_path,
        'stmap_w': stmap_w,
        'bvp_len': bvp_len,
        'hr_len': hr_len,
        'issues': issues,
        'ok': len(issues) == 0,
    }


def check_base(base: str, stmap_name: str, only_subject: str | None = None) -> dict:
    domains = region_domains(base)
    roots = {suffix: stmap_my_root(domains[suffix]) for suffix in REGION_SUFFIXES}

    missing_roots = [suffix for suffix, root in roots.items() if not os.path.isdir(root)]
    subject_sets = [set(list_subjects(roots[s])) for s in REGION_SUFFIXES if os.path.isdir(roots[s])]
    if not subject_sets:
        return {
            'base': base,
            'error': f'no region roots found under {base}',
            'missing_roots': missing_roots,
            'subjects': [],
        }

    common = set.intersection(*subject_sets) if subject_sets else set()
    union = set.union(*subject_sets)

    if only_subject is not None:
        subjects_to_check = [only_subject]
        missing_from_regions = [
            suffix for suffix in REGION_SUFFIXES
            if only_subject not in set(list_subjects(roots[suffix]))
        ]
        only_in_some_region = [only_subject] if missing_from_regions else []
    else:
        subjects_to_check = sorted(common)
        only_in_some_region = sorted(union - common)

    rows = []
    n_ok = 0
    n_fail = 0

    for subject_id in subjects_to_check:
        per_region: dict[str, dict] = {}
        row_issues: list[str] = []

        for suffix in REGION_SUFFIXES:
            root = roots[suffix]
            sub_root = os.path.join(root, subject_id)
            if not os.path.isdir(sub_root):
                row_issues.append(f'{suffix}: subject folder missing ({sub_root})')
                per_region[suffix] = {'ok': False, 'issues': ['subject folder missing']}
                continue
            info = check_subject_in_region(sub_root, stmap_name)
            per_region[suffix] = info
            row_issues.extend(f'{suffix}: {msg}' for msg in info['issues'])

        widths = {
            suffix: per_region[suffix]['stmap_w']
            for suffix in REGION_SUFFIXES
            if suffix in per_region and per_region[suffix].get('stmap_w') is not None
        }
        if len(widths) == 3 and len(set(widths.values())) > 1:
            row_issues.append(f'cross-region STMap width mismatch: {widths}')

        bvp_lens = {
            suffix: per_region[suffix]['bvp_len']
            for suffix in REGION_SUFFIXES
            if suffix in per_region and per_region[suffix].get('bvp_len') is not None
        }
        if len(bvp_lens) == 3 and len(set(bvp_lens.values())) > 1:
            row_issues.append(f'cross-region BVP length mismatch: {bvp_lens}')

        hr_lens = {
            suffix: per_region[suffix]['hr_len']
            for suffix in REGION_SUFFIXES
            if suffix in per_region and per_region[suffix].get('hr_len') is not None
        }
        if len(hr_lens) == 3 and len(set(hr_lens.values())) > 1:
            row_issues.append(f'cross-region HR length mismatch: {hr_lens}')

        ok = len(row_issues) == 0
        if ok:
            n_ok += 1
        else:
            n_fail += 1

        rows.append({
            'subject': subject_id,
            'ok': ok,
            'issues': row_issues,
            'per_region': per_region,
        })

    return {
        'base': base,
        'domains': domains,
        'roots': roots,
        'missing_roots': missing_roots,
        'stmap_name': stmap_name,
        'n_common_subjects': len(common) if only_subject is None else len(subjects_to_check),
        'n_union_subjects': len(union),
        'only_in_some_region': only_in_some_region,
        'n_ok': n_ok,
        'n_fail': n_fail,
        'rows': rows,
    }


def print_report(result: dict, verbose: bool = False) -> None:
    base = result['base']
    print('=' * 72)
    print(f'Base dataset: {base}')
    if 'error' in result:
        print('  ERROR:', result['error'])
        return

    if result['missing_roots']:
        print('  Missing region roots:', ', '.join(result['missing_roots']))

    for suffix in REGION_SUFFIXES:
        domain = result['domains'][suffix]
        print(f'  {suffix:4s} -> {domain}: {result["roots"][suffix]}')

    print(f'  STMap file: {STMAP_SUBDIR}/{result["stmap_name"]}')
    print(
        f'  Subjects in all 3 regions: {result["n_common_subjects"]}  '
        f'(union: {result["n_union_subjects"]})'
    )
    if result['only_in_some_region']:
        preview = result['only_in_some_region'][:8]
        extra = len(result['only_in_some_region']) - len(preview)
        tail = f' ... +{extra} more' if extra > 0 else ''
        print(f'  Not in all 3 regions ({len(result["only_in_some_region"])}): {preview}{tail}')

    print(f'  Ready: {result["n_ok"]}  |  Issues: {result["n_fail"]}')

    for row in result['rows']:
        if row['ok'] and not verbose:
            continue
        print('-' * 72)
        print(row['subject'])
        if row['ok']:
            in_info = row['per_region']['in']
            w = in_info.get('stmap_w')
            print(f'  OK  STMap/BVP/HR length = {w} (same across in/rm/eye)')
            continue
        for issue in row['issues']:
            print(f'  ! {issue}')
        if verbose:
            for suffix in REGION_SUFFIXES:
                info = row['per_region'].get(suffix, {})
                print(
                    f'    {suffix}: STMap={info.get("stmap_w")} '
                    f'BVP={info.get("bvp_len")} HR={info.get("hr_len")}'
                )

    print('=' * 72)


def run_check(
    bases: tuple[str, ...] | list[str],
    stmap_name: str = STMAP_NAME,
    only_subject: str | None = ONLY_SUBJECT,
    verbose: bool = VERBOSE,
) -> bool:
    """Run checks for all bases. Returns True if everything passed."""
    any_fail = False

    print('STMap_my root:', config.STMAP_DATA_ROOT)
    print('Checking bases:', ', '.join(bases))
    if only_subject:
        print('ONLY_SUBJECT:', only_subject)
    print()

    for base in bases:
        result = check_base(base, stmap_name, only_subject=only_subject)
        print_report(result, verbose=verbose)
        if result.get('n_fail', 0) > 0 or result.get('error'):
            any_fail = True
        elif not only_subject and result.get('only_in_some_region'):
            any_fail = True
        print()

    if any_fail:
        print('Some checks failed or subjects are missing from one or more regions.')
        return False
    print('All checked subjects are ready for train_regions (in/rm/eye lengths match).')
    return True


# %%
# --- Run check ---
run_check(CHECK_BASES, stmap_name=STMAP_NAME, only_subject=ONLY_SUBJECT, verbose=VERBOSE)

# %%
