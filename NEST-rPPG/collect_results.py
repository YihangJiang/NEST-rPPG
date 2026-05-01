"""
Aggregates all eval_result.json files under Wave_sort/ into a single CSV.
Columns: Source Domain, Target domain, Weight, Regions, Std, MAE, RMSE, r

Run after all train_regions.py + eval_from_bvp.py pairs have completed.
"""

import csv
import json
import os
import sys

import config

WAVE_SORT_ROOT = config.WAVE_SORT_ROOT
OUT_CSV = os.path.join(config.BASE_DIR, "all_results.csv")

FIELDNAMES = ["Source Domain", "Target domain", "Weight", "Regions", "Std", "MAE", "RMSE", "r"]

rows = []
for root, dirs, files in os.walk(WAVE_SORT_ROOT):
    for fname in files:
        if fname != "eval_result.json":
            continue
        json_path = os.path.join(root, fname)
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not read {json_path}: {e}", file=sys.stderr)
            continue

        src = data.get("source_domain", "")
        tgt = data.get("target_domain", "")
        weight = data.get("weight_info", "")
        result = data.get("result", {})

        for region, metrics in result.items():
            rows.append({
                "Source Domain": src,
                "Target domain": tgt,
                "Weight": weight,
                "Regions": region,
                "Std": metrics.get("Std", ""),
                "MAE": metrics.get("MAE", ""),
                "RMSE": metrics.get("RMSE", ""),
                "r": metrics.get("r", ""),
            })

rows.sort(key=lambda r: (r["Source Domain"], r["Target domain"], str(r["Weight"]), r["Regions"]))

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Collected {len(rows)} rows from {WAVE_SORT_ROOT}")
print(f"Saved: {OUT_CSV}")
