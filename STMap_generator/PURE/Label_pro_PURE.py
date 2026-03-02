# %%
"""
Generate the same Label files as NEST-rPPG/STMap/PURE/<session>/Label in PURE_my,
using only DATASET_3 (no copy from PURE).

Output: PURE_my/<session>/Label/Timestamp.mat, HR.mat, SPO2.mat, BVP.mat, BVP_Filt.mat.
Source: DATASET_3 (e.g. /mnt/nvme2/DATASET_3/<session>/<session>.json and .../<session>/Image<ts>.png).
- JSON /FullPackage[]: Timestamp, Value.pulseRate (HR), Value.o2saturation (SPO2), Value.waveform (BVP).
- Frame timestamps from PNG filenames; signals interpolated to frame times.
- BVP/BVP_Filt: normalize to [0,1]; BVP_Filt = butter bandpass (0.7–2.5 Hz) then normalize.

Run after Landmark_PURE.py (PURE_my/<sub>/Label/RGB_lmk.csv must exist for frame count).
"""
import os
import re
import json
import numpy as np
import scipy.io as scio
from scipy import interpolate
from scipy import signal

# %%
# Same filter as BUAA Label_pro.py (40–240 bpm bandpass for PPG)
LPF = 0.7   # Hz (~40 bpm)
HPF = 2.5   # Hz (~240 bpm)
NyquistF = 15
FS = 30
B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')

# Raw PURE dataset (DATASET_3); override if your path differs
DATASET_3_ROOT = '/mnt/nvme2/rppg_data/DATASET_3'
_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_script_dir, '..', '..'))
PURE_MY_ROOT = os.path.join(PROJECT_ROOT, 'STMap_my', 'PURE_my')

# %%
def _list_session_structure(session_path, max_entries=50):
    """Print directory structure under session_path for debugging."""
    if not os.path.isdir(session_path):
        return f"  [not a dir] {session_path}"
    lines = []
    for entry in sorted(os.listdir(session_path))[:max_entries]:
        full = os.path.join(session_path, entry)
        if os.path.isdir(full):
            nfiles = len([f for f in os.listdir(full) if f.endswith('.png')])
            lines.append(f"  {entry}/ ({nfiles} .png)")
        else:
            lines.append(f"  {entry}")
    if len(os.listdir(session_path)) > max_entries:
        lines.append(f"  ... ({len(os.listdir(session_path))} total)")
    return "\n".join(lines) if lines else "  (empty)"


def get_frame_timestamps_from_png(session_path):
    """Get sorted frame timestamps from DATASET_3 session PNG filenames.
    Tries: Image<ts>.png (PURE standard), then <digits>.png, in <session>/<session>/,
    then <session>/, then any subdir under <session>.
    """
    session_name = os.path.basename(session_path.rstrip(os.sep))
    patterns = [
        (re.compile(r'Image(\d+)\.png', re.IGNORECASE), True),   # Image123456.png -> ts
        (re.compile(r'^(\d+)\.png$'), True),                      # 000001.png -> treat as ts or order
    ]
    timestamps = []

    def collect_from_dir(d):
        if not os.path.isdir(d):
            return
        for f in os.listdir(d):
            for pat, _ in patterns:
                m = pat.match(f)
                if m:
                    timestamps.append(int(m.group(1)))
                    break

    # 1) Nested folder then session root
    for candidate_dir in [os.path.join(session_path, session_name), session_path]:
        collect_from_dir(candidate_dir)
        if timestamps:
            return sorted(timestamps)

    # 2) Any subdirectory
    if os.path.isdir(session_path):
        for entry in sorted(os.listdir(session_path)):
            subdir = os.path.join(session_path, entry)
            if os.path.isdir(subdir):
                collect_from_dir(subdir)
        if timestamps:
            return sorted(timestamps)

    return sorted(timestamps) if timestamps else None


def load_json_signals(json_path):
    """Load /FullPackage from PURE session JSON. Returns (ts, hr, spo2, waveform) or None."""
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:
        return None
    pack = data.get('/FullPackage', data.get('FullPackage', []))
    if not pack:
        return None
    ts = [int(x['Timestamp']) for x in pack if 'Timestamp' in x]
    hr = [float(x['Value']['pulseRate']) for x in pack if 'Value' in x and 'pulseRate' in x['Value']]
    spo2 = [float(x['Value']['o2saturation']) for x in pack if 'Value' in x and 'o2saturation' in x['Value']]
    waveform = [float(x['Value']['waveform']) for x in pack if 'Value' in x and 'waveform' in x['Value']]
    if not (len(ts) == len(hr) == len(spo2) == len(waveform)):
        return None
    return np.array(ts), np.array(hr), np.array(spo2), np.array(waveform)


def interpolate_to_times(t_orig, y_orig, t_target):
    """Spline interpolate y_orig(t_orig) at t_target. Clamp to valid range."""
    t_orig = np.asarray(t_orig, dtype=np.float64)
    y_orig = np.asarray(y_orig, dtype=np.float64)
    t_target = np.asarray(t_target, dtype=np.float64)
    if len(t_orig) < 4:
        return np.full_like(t_target, np.nanmean(y_orig))
    t_min, t_max = t_orig.min(), t_orig.max()
    t_target = np.clip(t_target, t_min, t_max)
    spl = interpolate.splrep(t_orig, y_orig, k=min(3, len(t_orig) - 1))
    return interpolate.splev(t_target, spl)


# %%
if not os.path.isdir(PURE_MY_ROOT):
    print(f"PURE_my root not found: {PURE_MY_ROOT}")
else:
    for sub_name in sorted(os.listdir(PURE_MY_ROOT)):
        if sub_name.startswith('.'):
            continue
        pure_my_label = os.path.join(PURE_MY_ROOT, sub_name, 'Label')
        lmk_csv = os.path.join(pure_my_label, 'RGB_lmk.csv')
        if not os.path.isfile(lmk_csv):
            print(f"  Skip (no RGB_lmk.csv): {sub_name}")
            continue

        # Frame count from landmarks
        with open(lmk_csv) as f:
            n_lmk = sum(1 for _ in f)

        session_path = os.path.join(DATASET_3_ROOT, sub_name)
        json_path = os.path.join(session_path, sub_name + '.json')
        signals = load_json_signals(json_path)
        if signals is None:
            print(f"  Skip (no/invalid JSON): {json_path}")
            continue
        ts_json, hr_json, spo2_json, wave_json = signals

        # Frame timestamps: from PNG filenames, or fallback to JSON timestamps (first n_lmk)
        frame_ts = get_frame_timestamps_from_png(session_path)
        if frame_ts is None or len(frame_ts) == 0:
            # Fallback: use JSON timestamps (first n_lmk) when no Image*.png found
            if len(ts_json) >= n_lmk:
                frame_ts = np.array(ts_json[:n_lmk], dtype=np.float64)
                HR = np.array(hr_json[:n_lmk], dtype=np.float64)
                SPO2 = np.array(spo2_json[:n_lmk], dtype=np.float64)
                waveform = np.array(wave_json[:n_lmk], dtype=np.float64)
                n = n_lmk
                if sub_name == "10-06" or os.environ.get("PURE_LABEL_DEBUG"):
                    print(f"  {sub_name}: using JSON timestamps (no PNG found). Structure:")
                    print(_list_session_structure(session_path))
            else:
                print(f"  Skip (no PNG timestamps, JSON has {len(ts_json)} < n_lmk={n_lmk}): {sub_name}")
                if sub_name == "10-06" or os.environ.get("PURE_LABEL_DEBUG"):
                    print(f"  DATASET_3 structure for {session_path}:")
                    print(_list_session_structure(session_path))
                continue
        else:
            frame_ts = np.array(frame_ts, dtype=np.float64)
            HR = interpolate_to_times(ts_json, hr_json, frame_ts)
            SPO2 = interpolate_to_times(ts_json, spo2_json, frame_ts)
            waveform = interpolate_to_times(ts_json, wave_json, frame_ts)
            n = min(len(frame_ts), n_lmk)
            frame_ts = frame_ts[:n]
            HR = HR[:n]
            SPO2 = SPO2[:n]
            waveform = waveform[:n]
            if len(frame_ts) != n_lmk:
                print(f"  {sub_name}: frame_ts={len(frame_ts)}, landmark rows={n_lmk}")

        n = len(frame_ts)

        os.makedirs(pure_my_label, exist_ok=True)

        # Timestamp.mat: (1, N) int64 like NEST PURE
        scio.savemat(os.path.join(pure_my_label, 'Timestamp.mat'),
                     {'Timestamp': np.array(frame_ts, dtype=np.int64).reshape(1, -1)})
        # HR.mat, SPO2.mat: (1, N) float64
        scio.savemat(os.path.join(pure_my_label, 'HR.mat'), {'HR': HR.reshape(1, -1)})
        scio.savemat(os.path.join(pure_my_label, 'SPO2.mat'), {'SPO2': SPO2.reshape(1, -1)})

        # BVP: normalize to [0,1] (same as BUAA)
        pulse_csi = np.array(waveform, dtype=np.float64)
        pmin, pmax = np.nanmin(pulse_csi), np.nanmax(pulse_csi)
        if pmax > pmin:
            pulse_csi = (pulse_csi - pmin) / (pmax - pmin)
        else:
            pulse_csi = np.zeros_like(pulse_csi)
        scio.savemat(os.path.join(pure_my_label, 'BVP.mat'), {'BVP': pulse_csi.reshape(1, -1)})

        # BVP_Filt: bandpass then normalize (same as BUAA Label_pro)
        EXG2_f = signal.filtfilt(B, A, pulse_csi)
        fmin, fmax = np.nanmin(EXG2_f), np.nanmax(EXG2_f)
        if fmax > fmin:
            EXG2_f = (EXG2_f - fmin) / (fmax - fmin)
        else:
            EXG2_f = np.zeros_like(EXG2_f)
        scio.savemat(os.path.join(pure_my_label, 'BVP_Filt.mat'), {'BVP': EXG2_f.reshape(1, -1)})

        print(sub_name, '->', pure_my_label, f'({n} frames)')

# %%