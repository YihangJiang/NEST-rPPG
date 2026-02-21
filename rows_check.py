# %%
#!/usr/bin/env python3

"""
Scan landmark CSVs in STMap_my/UBFC_my and NEST-rPPG/STMap/UBFC and report which files
have rows of all zeros (no face detected for that frame).
"""
import os
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UBFC_MY_ROOT = os.path.join(SCRIPT_DIR, "STMap_my", "UBFC_my")
# NEST-rPPG/STMap/UBFC (reference/original UBFC STMap folder)
STMAP_UBFC_ROOT = os.path.join(SCRIPT_DIR, "NEST-rPPG", "STMap", "UBFC")


def is_zero_row(values):
    """True if the row is all zeros (or 0.0)."""
    try:
        return all(float(x) == 0 for x in values)
    except (ValueError, TypeError):
        return False


def check_csv(path):
    """
    Read CSV and return (total_rows, zero_row_indices).
    """
    zero_indices = []
    total = 0
    with open(path, "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            total += 1
            if is_zero_row(row):
                zero_indices.append(i)
    return total, zero_indices


def check_root(root_path):
    """Check all Label/RGB_lmk.csv under root_path and print all-zero row report."""
    if not os.path.isdir(root_path):
        print(f"Directory not found: {root_path}")
        return
    for sub_name in sorted(os.listdir(root_path)):
        if sub_name.startswith("."):
            continue
        csv_path = os.path.join(root_path, sub_name, "Label", "RGB_lmk.csv")
        if not os.path.isfile(csv_path):
            continue

        total, zero_indices = check_csv(csv_path)
        n_zero = len(zero_indices)

        if n_zero == 0:
            print(f"  {sub_name}: {total} rows, no all-zero rows")
        else:
            pct = 100.0 * n_zero / total if total else 0
            print(f"  {sub_name}: {total} rows, {n_zero} all-zero rows ({pct:.1f}%)")
            if n_zero <= 20:
                print(f"    zero row indices (0-based): {zero_indices}")
            else:
                print(f"    zero row indices (0-based): {zero_indices[:10]} ... {zero_indices[-5:]} (first 10, last 5)")


def main():
    # 1) STMap_my/UBFC_my
    print("Checking landmark CSVs in STMap_my/UBFC_my for all-zero rows (no face detected).")
    print("=" * 70)
    check_root(UBFC_MY_ROOT)
    print()

    # 2) NEST-rPPG/STMap/UBFC
    print("Checking landmark CSVs in NEST-rPPG/STMap/UBFC for all-zero rows (no face detected).")
    print("=" * 70)
    check_root(STMAP_UBFC_ROOT)
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()

# %%
