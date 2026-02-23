# %%
"""
Compare Label .mat files (and optionally landmarks) between NEST-rPPG/STMap/PURE and STMap_my/PURE_my.
Reports whether PURE_my and PURE labels match (per session and overall).
Generates overlay and difference figures per session for BVP, BVP_Filt, HR, SPO2.
"""
import os
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# Paths: script lives in analysis/, project root is parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Roots
NEST_PURE_ROOT = os.path.join(PROJECT_ROOT, "NEST-rPPG", "STMap", "PURE")
PURE_MY_ROOT = os.path.join(PROJECT_ROOT, "STMap_my", "PURE_my")

# Tolerance for "match": max allowed MAE for float signals; timestamps must match exactly or within 1
MATCH_MAE_TOLERANCE = 1e-5
TIMESTAMP_TOLERANCE = 1  # max allowed abs diff for timestamp (0 = exact)


# %%
def load_mat_signal(path, key, dtype=float):
    """Load 1D signal from .mat file key; return None on error."""
    if not os.path.isfile(path):
        return None
    try:
        data = scio.loadmat(path)
        if key not in data:
            return None
        return np.squeeze(data[key]).astype(dtype)
    except Exception:
        return None


def compare_signal_pair(sig_a, sig_b, name, tolerance_mae=MATCH_MAE_TOLERANCE, exact_int=False):
    """
    Compare two 1D signals. Return (mae, rmse, max_diff, n_common, match).
    match = True if lengths equal and MAE <= tolerance (or exact match for int).
    """
    if sig_a is None or sig_b is None:
        return None, None, None, 0, False
    n_a, n_b = len(sig_a), len(sig_b)
    n = min(n_a, n_b)
    a, b = sig_a[:n], sig_b[:n]
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_d = float(np.max(diff))
    if exact_int:
        match = n_a == n_b and np.allclose(a, b, atol=0)
    else:
        match = n_a == n_b and mae <= tolerance_mae
    return mae, rmse, max_d, n, match


def compare_session(session_name):
    """Compare one session's Label between NEST PURE and PURE_my. Return dict of results."""
    nest_label = os.path.join(NEST_PURE_ROOT, session_name, "Label")
    my_label = os.path.join(PURE_MY_ROOT, session_name, "Label")
    results = {"session": session_name, "files": {}, "all_match": True}

    # Timestamp.mat (int64)
    ts_nest = load_mat_signal(os.path.join(nest_label, "Timestamp.mat"), "Timestamp", dtype=np.int64)
    ts_my = load_mat_signal(os.path.join(my_label, "Timestamp.mat"), "Timestamp", dtype=np.int64)
    mae, rmse, max_d, n, match = compare_signal_pair(
        ts_nest, ts_my, "Timestamp", tolerance_mae=TIMESTAMP_TOLERANCE, exact_int=(TIMESTAMP_TOLERANCE == 0)
    )
    results["files"]["Timestamp"] = {"mae": mae, "rmse": rmse, "max_diff": max_d, "n": n, "match": match}
    if not match:
        results["all_match"] = False

    # HR.mat, SPO2.mat, BVP.mat, BVP_Filt.mat (float) — file stem and .mat key
    for file_stem, key in [("HR", "HR"), ("SPO2", "SPO2"), ("BVP", "BVP"), ("BVP_Filt", "BVP")]:
        path_nest = os.path.join(nest_label, f"{file_stem}.mat")
        path_my = os.path.join(my_label, f"{file_stem}.mat")
        a = load_mat_signal(path_nest, key)
        b = load_mat_signal(path_my, key)
        mae, rmse, max_d, n, match = compare_signal_pair(a, b, file_stem)
        results["files"][file_stem] = {"mae": mae, "rmse": rmse, "max_diff": max_d, "n": n, "match": match}
        if mae is not None and not match:
            results["all_match"] = False

    return results


def plot_signal_comparison(signal_a, signal_b, session_name, signal_type, save_dir, ylabel=None):
    """Create overlay and difference plots for two 1D signals. Returns (overlay_path, diff_path)."""
    if signal_a is None or signal_b is None:
        return None, None
    n = min(len(signal_a), len(signal_b))
    a = np.asarray(signal_a[:n], dtype=float)
    b = np.asarray(signal_b[:n], dtype=float)
    t = np.arange(n)
    safe_name = session_name.replace("/", "-")
    ylabel = ylabel or signal_type

    # Overlay
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, a, label="NEST-rPPG/STMap/PURE", alpha=0.7, linewidth=1)
    ax1.plot(t, b, label="PURE_my", alpha=0.7, linewidth=1)
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"{session_name} - {signal_type} Overlay")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    overlay_path = os.path.join(save_dir, f"{safe_name}_{signal_type}_overlay.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Difference
    diff = np.abs(a - b)
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(t, diff, label="|A - B|", color="red", alpha=0.7, linewidth=1)
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("|A - B|")
    mae, rmse = float(np.mean(diff)), float(np.sqrt(np.mean(diff ** 2)))
    ax2.set_title(f"{session_name} - {signal_type} Difference (MAE={mae:.6f}, RMSE={rmse:.6f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    diff_path = os.path.join(save_dir, f"{safe_name}_{signal_type}_diff.png")
    plt.savefig(diff_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    return overlay_path, diff_path


def load_lmk_csv(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(",")]
            rows.append(vals)
    return np.array(rows, dtype=float)


# %%
print("=" * 60)
print("PURE label comparison: NEST-rPPG/STMap/PURE vs STMap_my/PURE_my")
print("=" * 60)

# Collect common session names (folder names in both roots)
nest_folders = set()
if os.path.isdir(NEST_PURE_ROOT):
    for name in os.listdir(NEST_PURE_ROOT):
        if not name.startswith(".") and os.path.isdir(os.path.join(NEST_PURE_ROOT, name)):
            nest_folders.add(name)

my_folders = set()
if os.path.isdir(PURE_MY_ROOT):
    for name in os.listdir(PURE_MY_ROOT):
        if not name.startswith(".") and os.path.isdir(os.path.join(PURE_MY_ROOT, name)):
            my_folders.add(name)

common = sorted(nest_folders & my_folders)
only_nest = sorted(nest_folders - my_folders)
only_my = sorted(my_folders - nest_folders)

print(f"\nCommon sessions: {len(common)}")
print(f"Only in NEST PURE: {len(only_nest)}")
print(f"Only in PURE_my: {len(only_my)}")
if only_nest:
    print(f"  Only NEST: {only_nest[:15]}{'...' if len(only_nest) > 15 else ''}")
if only_my:
    print(f"  Only PURE_my: {only_my[:15]}{'...' if len(only_my) > 15 else ''}")

# Part 1: Landmarks (only when both have RGB_lmk.csv)
print("\n" + "-" * 60)
print("PART 1: Landmarks (RGB_lmk.csv)")
print("-" * 60)
lmk_compared = 0
lmk_match_count = 0
for name in common:
    nest_lmk = os.path.join(NEST_PURE_ROOT, name, "Label", "RGB_lmk.csv")
    my_lmk = os.path.join(PURE_MY_ROOT, name, "Label", "RGB_lmk.csv")
    if not os.path.isfile(nest_lmk):
        continue
    if not os.path.isfile(my_lmk):
        print(f"  {name}: PURE_my missing RGB_lmk.csv")
        continue
    lmk_a = load_lmk_csv(nest_lmk)
    lmk_b = load_lmk_csv(my_lmk)
    n = min(lmk_a.shape[0], lmk_b.shape[0])
    diff = np.abs(lmk_a[:n] - lmk_b[:n])
    mae = np.mean(diff)
    match = lmk_a.shape[0] == lmk_b.shape[0] and mae < 1e-6
    lmk_compared += 1
    if match:
        lmk_match_count += 1
    else:
        print(f"  {name}: rows NEST={lmk_a.shape[0]} PURE_my={lmk_b.shape[0]}, MAE={mae:.6f} {'MATCH' if match else 'MISMATCH'}")
if lmk_compared == 0:
    print("  No pairs with both RGB_lmk.csv found (NEST PURE may not have landmarks).")
else:
    print(f"  Compared {lmk_compared} sessions; {lmk_match_count} landmark match.")

# Part 2: Label .mat files (Timestamp, HR, SPO2, BVP, BVP_Filt) and figures
print("\n" + "-" * 60)
print("PART 2: Label .mat files (Timestamp, HR, SPO2, BVP, BVP_Filt)")
print("-" * 60)

FIGURES_DIR = os.path.join(SCRIPT_DIR, "pure_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"Figures will be saved to: {FIGURES_DIR}")

# Signals to plot (file_stem, .mat key, ylabel)
PLOT_SIGNALS = [
    ("HR", "HR", "HR"),
    ("SPO2", "SPO2", "SPO2"),
    ("BVP", "BVP", "BVP"),
    ("BVP_Filt", "BVP", "BVP_Filt"),
]

session_results = []
for name in common:
    r = compare_session(name)
    session_results.append(r)
    if not r["all_match"]:
        print(f"\n  {name}: MISMATCH")
        for fname, stats in r["files"].items():
            if stats["mae"] is not None and not stats["match"]:
                print(f"    {fname}: n={stats['n']}, MAE={stats['mae']:.6f}, max_diff={stats['max_diff']:.6f}")

    # Generate figures for this session (overlay + diff per signal)
    nest_label = os.path.join(NEST_PURE_ROOT, name, "Label")
    my_label = os.path.join(PURE_MY_ROOT, name, "Label")
    fig_count = 0
    for file_stem, key, ylabel in PLOT_SIGNALS:
        path_nest = os.path.join(nest_label, f"{file_stem}.mat")
        path_my = os.path.join(my_label, f"{file_stem}.mat")
        a = load_mat_signal(path_nest, key)
        b = load_mat_signal(path_my, key)
        if a is not None and b is not None:
            ov, df = plot_signal_comparison(a, b, name, file_stem, FIGURES_DIR, ylabel=ylabel)
            if ov and df:
                fig_count += 2
    if fig_count:
        print(f"  {name}: saved {fig_count} figures")

# Summary
print("\n" + "=" * 60)
print("SUMMARY: Do PURE_my and PURE label match?")
print("=" * 60)

all_match_sessions = sum(1 for r in session_results if r["all_match"])
total = len(session_results)
print(f"  Sessions compared: {total}")
print(f"  Sessions with all signals matching (MAE <= {MATCH_MAE_TOLERANCE}): {all_match_sessions}")

if total == 0:
    print("\n  No common sessions to compare. Check NEST-rPPG/STMap/PURE and STMap_my/PURE_my folder names.")
elif all_match_sessions == total:
    print("\n  YES – PURE_my and PURE labels match for all compared sessions.")
else:
    print(f"\n  NO – {total - all_match_sessions} session(s) have at least one label file that does not match.")
    print("  Mismatched sessions listed above in Part 2.")

print("=" * 60)

# %%
