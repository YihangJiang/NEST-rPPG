# %%
"""
Find the mapping between NEST-rPPG/STMap/PURE folder names (10000, 10003, ...)
and DATASET_3 (PURE raw) folder names (01-01, 01-02, ...) using timestamps.

- NEST side: each subject folder has Label/Timestamp.mat (one timestamp per frame).
- DATASET_3 side: each session has a subfolder with PNGs named Image<timestamp>.png.
Matching: first timestamp or best overlap. Run cells in order in Jupyter or VS Code.
"""
import os
import re
import json

try:
    import scipy.io as scio
except ImportError:
    scio = None

# %%
# Config: edit paths and run this cell first
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..'))
NEST_PURE_ROOT = os.path.join(PROJECT_ROOT, 'NEST-rPPG', 'STMap', 'PURE')
DATASET_3_ROOT = '/mnt/nvme2/rppg_data/DATASET_3'
TOLERANCE_NS = 10_000_000  # 10 ms for first-timestamp match
OUT_CSV = None  # set to a path, e.g. 'pure_nest_to_dataset3_mapping.csv', to save mapping

# %%
def get_nest_timestamps(nest_subject_path):
    """Load Label/Timestamp.mat for a NEST subject. Returns sorted list of int timestamps or None."""
    mat_path = os.path.join(nest_subject_path, 'Label', 'Timestamp.mat')
    if not os.path.isfile(mat_path) or scio is None:
        return None
    try:
        data = scio.loadmat(mat_path)
        if 'Timestamp' not in data:
            return None
        ts = data['Timestamp'].ravel()
        return sorted([int(t) for t in ts])
    except Exception:
        return None


def get_dataset3_timestamps_from_png(session_path):
    """Get sorted frame timestamps from DATASET_3 by parsing PNG filenames Image<ts>.png."""
    session_name = os.path.basename(session_path.rstrip(os.sep))
    frames_dir = os.path.join(session_path, session_name)
    if not os.path.isdir(frames_dir):
        frames_dir = session_path
    if not os.path.isdir(frames_dir):
        return None
    pattern = re.compile(r'Image(\d+)\.png', re.IGNORECASE)
    timestamps = []
    for f in os.listdir(frames_dir):
        m = pattern.match(f)
        if m:
            timestamps.append(int(m.group(1)))
    return sorted(timestamps) if timestamps else None


def get_dataset3_timestamps_from_json(session_path):
    """Fallback: get timestamps from session JSON /FullPackage[].Timestamp."""
    session_name = os.path.basename(session_path.rstrip(os.sep))
    json_path = os.path.join(session_path, session_name + '.json')
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
        pack = data.get('/FullPackage', data.get('FullPackage', []))
        if not pack:
            return None
        return sorted([int(x['Timestamp']) for x in pack if 'Timestamp' in x])
    except Exception:
        return None


def match_by_first_ts(nest_ts_list, dataset_ts_list, tolerance_ns=10_000_000):
    """True if the first timestamps match within tolerance."""
    if not nest_ts_list or not dataset_ts_list:
        return False
    return abs(nest_ts_list[0] - dataset_ts_list[0]) <= tolerance_ns


def overlap_count(ts_a, ts_b, tolerance_ns=0):
    """Count how many of ts_a appear in ts_b (set intersection when tolerance_ns==0)."""
    if tolerance_ns == 0:
        set_b = set(ts_b)
        return sum(1 for t in ts_a if t in set_b)
    count = 0
    for t in ts_a:
        for tb in ts_b:
            if abs(t - tb) <= tolerance_ns:
                count += 1
                break
    return count

# %%
# Collect NEST subjects and DATASET_3 sessions
nest_root = os.path.abspath(NEST_PURE_ROOT)
dataset_root = os.path.abspath(DATASET_3_ROOT)
nest_subjects = []
dataset_sessions = []

if not os.path.isdir(nest_root):
    print(f"NEST PURE root not found: {nest_root}")
elif not os.path.isdir(dataset_root):
    print(f"DATASET_3 root not found: {dataset_root}")
elif scio is None:
    print("scipy is required. Install with: pip install scipy")
else:
    for name in sorted(os.listdir(nest_root)):
        if name.startswith('.'):
            continue
        path = os.path.join(nest_root, name)
        if not os.path.isdir(path):
            continue
        ts = get_nest_timestamps(path)
        if ts is not None:
            nest_subjects.append((name, ts))
        else:
            print(f"  [NEST] No timestamps for: {name}")

    for name in sorted(os.listdir(dataset_root)):
        if name.startswith('.'):
            continue
        path = os.path.join(dataset_root, name)
        if not os.path.isdir(path):
            continue
        ts = get_dataset3_timestamps_from_png(path)
        if ts is not None:
            dataset_sessions.append((name, ts))
        else:
            ts_json = get_dataset3_timestamps_from_json(path)
            if ts_json is not None:
                dataset_sessions.append((name, ts_json))

    print(f"Found {len(nest_subjects)} NEST subjects and {len(dataset_sessions)} DATASET_3 sessions.")

# %%
# Compute mapping: for each NEST subject, find best DATASET_3 session (first_ts or overlap)
tolerance = TOLERANCE_NS
mapping = []  # (nest_id, dataset3_id, match_method, overlap_info)

for nest_id, nest_ts in nest_subjects:
    best_session = None
    best_first_match = False
    best_overlap = -1
    best_len_ratio = 0

    first_ts = nest_ts[0]
    candidates = [(ds_id, ds_ts) for ds_id, ds_ts in dataset_sessions
                  if ds_ts and abs(ds_ts[0] - first_ts) <= tolerance]
    if not candidates:
        candidates = [(ds_id, ds_ts) for ds_id, ds_ts in dataset_sessions if ds_ts]

    for ds_id, ds_ts in candidates:
        first_ok = match_by_first_ts(nest_ts, ds_ts, tolerance)
        overlap = overlap_count(nest_ts, ds_ts, tolerance_ns=0)
        len_ratio = len(nest_ts) / len(ds_ts) if ds_ts else 0
        if first_ok and (best_session is None or overlap > best_overlap):
            best_session = ds_id
            best_first_match = True
            best_overlap = overlap
            best_len_ratio = len_ratio
        elif not first_ok and (best_session is None or overlap > best_overlap):
            best_session = ds_id
            best_first_match = False
            best_overlap = overlap
            best_len_ratio = len_ratio

    if best_session is None:
        print(f"  {nest_id} -> (no match)")
        mapping.append((nest_id, "", "no_match", ""))
        continue

    overlap_info = f"overlap={best_overlap}/{len(nest_ts)} frames, len_ratio={best_len_ratio:.3f}"
    method = "first_ts" if best_first_match else "overlap"
    if best_overlap == 0:
        print(f"  {nest_id} -> {best_session}  [{method}] {overlap_info}  [WARNING: zero overlap]")
    else:
        print(f"  {nest_id} -> {best_session}  [{method}] {overlap_info}")
    mapping.append((nest_id, best_session, method, overlap_info))

# %%
# Warnings and optional CSV save
ds_to_nest = {}
for nest_id, ds_id, _, _ in mapping:
    if ds_id:
        ds_to_nest.setdefault(ds_id, []).append(nest_id)
for ds_id, nest_ids in ds_to_nest.items():
    if len(nest_ids) > 1:
        print(f"  [Note] DATASET_3 {ds_id} mapped by multiple NEST ids: {nest_ids}")

if OUT_CSV:
    out_path = os.path.join(SCRIPT_DIR, OUT_CSV) if not os.path.isabs(OUT_CSV) else OUT_CSV
    with open(out_path, 'w') as f:
        f.write("nest_id,dataset3_id,match_method,overlap_info\n")
        for row in mapping:
            f.write(",".join([str(x) for x in row]) + "\n")
    print(f"\nMapping written to: {out_path}")


# %%
# Load Timestamp.mat for a specific NEST subject (e.g. 10060)
NEST_SUBJECT_ID = '10060'  # change to load another subject
ts_mat_10060 = None
ts_list_10060 = None
if scio is not None:
    nest_root = os.path.abspath(NEST_PURE_ROOT)
    mat_path = os.path.join(nest_root, NEST_SUBJECT_ID, 'Label', 'Timestamp.mat')
    if os.path.isfile(mat_path):
        ts_mat_10060 = scio.loadmat(mat_path)  # full .mat dict (e.g. 'Timestamp' array)
        ts_list_10060 = ts_mat_10060['Timestamp'].ravel()
        ts_list_10060 = sorted([int(t) for t in ts_list_10060])
        print(f"Subject {NEST_SUBJECT_ID}: {len(ts_list_10060)} frames, first={ts_list_10060[0]}, last={ts_list_10060[-1]}")
    else:
        print(f"Not found: {mat_path}")
else:
    print("scipy not available; cannot load .mat")
# Use ts_mat_10060 (raw .mat) or ts_list_10060 (sorted list of int timestamps)
# %%
