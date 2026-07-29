#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Script to search, inspect, and summarize Optuna databases in Training_Log/optuna/.
"""
# %%
import os
import glob
import optuna

import config
def inspect_optuna_databases(optuna_dir=None):
    if optuna_dir is None:
        optuna_dir = os.path.join(config.RESULT_LOG_DIR, "optuna")

    if not os.path.isdir(optuna_dir):
        print(f"[ERROR] Directory not found: {optuna_dir}")
        return

    db_files = sorted(glob.glob(os.path.join(optuna_dir, "*.db")))
    if not db_files:
        print(f"No Optuna SQLite database (.db) files found in: {optuna_dir}")
        return

    print("=" * 80)
    print(f"Found {len(db_files)} Optuna database file(s) in: {optuna_dir}")
    print("=" * 80)

    for db_path in db_files:
        db_name = os.path.basename(db_path)
        storage_uri = f"sqlite:///{os.path.abspath(db_path)}"
        
        try:
            summaries = optuna.study.get_all_study_summaries(storage=storage_uri)
        except Exception as e:
            print(f"\n[Warning] Failed to read {db_name}: {e}")
            continue

        print(f"\n📁 Database File: {db_name}")
        print("-" * 80)

        for summary in summaries:
            study_name = summary.study_name
            study = optuna.load_study(study_name=study_name, storage=storage_uri)
            
            print(f"  Study Name  : {study_name}")
            print(f"  Total Trials: {len(study.trials)} (Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])})")
            
            if study.best_trial is not None:
                print(f"  ⭐ Best Trial #: {study.best_trial.number}")
                print(f"  ⭐ Best MAE    : {study.best_value:.4f}")
                print(f"  ⭐ Best Params : {study.best_params}")
                if "wave_sort_path" in study.best_trial.user_attrs:
                    print(f"     Wave_sort  : {study.best_trial.user_attrs['wave_sort_path']}")
            else:
                print("  No completed trials found in this study.")
            print("-" * 80)

if __name__ == "__main__":
    inspect_optuna_databases()
