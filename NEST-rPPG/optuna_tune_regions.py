#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Optuna hyperparameter search for train_regions.py.

Searches: tau_info, weight_info from config sweeps (loss_type fixed to config.LOSS_TYPE).
Objective: minimize eval HR MAE (via utils.eval_utils.run_eval).

Example:
  python optuna_tune_regions.py --src PURE_my_rm -t UBFC_my_rm --regions pos --n-trials 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "optuna is required. Install with: pip install optuna"
    ) from exc

import config
from utils.eval_utils import run_eval

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_WAVE_SORT_PATH = os.path.join(config.RESULT_LOG_DIR, "last_wave_sort_path.txt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna fine-tuning for train_regions (tau_info, weight_info)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", required=True, help="Source domain (e.g. PURE_my_rm)")
    parser.add_argument("-t", "--tgt", required=True, help="Target domain (e.g. UBFC_my_rm)")
    parser.add_argument(
        "--regions",
        default="all",
        choices=["all", "neg", "pos"],
        help="InfoNCE region mode passed to train_regions.py",
    )
    parser.add_argument("-g", "--gpu", default="0", help="GPU index for training")
    parser.add_argument("-s", "--seed", type=int, default=config.SEED, help="Training seed")
    parser.add_argument("-mi", "--max-iter", type=int, default=None, help="Training iterations")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name (default: regions_<src>_<tgt>)",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URI (default: sqlite under Training_Log/optuna/)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Max wall-clock seconds for the whole study",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete saved study DB for this study name and start over",
    )
    return parser.parse_args()


def _search_space_tag() -> str:
    """Unique id for current tau/weight sweeps (avoids loading incompatible old trials)."""
    payload = json.dumps(
        {
            "tau": list(config.OPTUNA_TAU_INFO_SWEEP),
            "weight": list(config.WEIGHT_INFO_SWEEP),
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def _grid_sampler() -> optuna.samplers.GridSampler:
    return optuna.samplers.GridSampler(
        {
            "tau_info": list(config.OPTUNA_TAU_INFO_SWEEP),
            "weight_info": list(config.WEIGHT_INFO_SWEEP),
        }
    )


def _default_study_name(src: str, tgt: str, regions: str) -> str:
    safe = lambda s: s.replace("/", "_")
    return f"regions_{safe(src)}_{safe(tgt)}_{regions}_{_search_space_tag()}"


def _default_storage(study_name: str) -> str:
    optuna_dir = os.path.join(config.RESULT_LOG_DIR, "optuna")
    os.makedirs(optuna_dir, exist_ok=True)
    db_path = os.path.join(optuna_dir, f"{study_name}.db")
    return "sqlite:///" + os.path.abspath(db_path)


def _run_training(
    *,
    src: str,
    tgt: str,
    regions: str,
    weight_info: float,
    tau_info: float,
    run_suffix: str,
    gpu: str,
    seed: int,
    max_iter: int | None,
) -> None:
    cmd = [
        sys.executable,
        "train_regions.py",
        "--src",
        src,
        "-t",
        tgt,
        "--regions",
        regions,
        "--weight_info",
        str(weight_info),
        "--tau-info",
        str(tau_info),
        "-g",
        gpu,
        "-s",
        str(seed),
        "--mlflow-run-name",
        f"optuna_{run_suffix}",
    ]
    if max_iter is not None:
        cmd.extend(["-mi", str(max_iter)])

    print("\n" + "=" * 60)
    print("Optuna trial command:")
    print(" ", " ".join(cmd))
    print(f"  weight_info={weight_info}  tau_info={tau_info:.6f}  loss_type={config.LOSS_TYPE}")
    print("Training progress (updates every 100 iters):")
    print("=" * 60)
    sys.stdout.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NEST_TRAIN_REGIONS_RUN_SUFFIX"] = run_suffix
    result = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"train_regions.py failed (exit {result.returncode}). "
            f"Check Training_Log/rPPGNet_*_{run_suffix}_log.txt"
        )


def _read_wave_sort_path() -> str:
    if not os.path.isfile(LAST_WAVE_SORT_PATH):
        raise FileNotFoundError(f"Missing wave sort pointer: {LAST_WAVE_SORT_PATH}")
    with open(LAST_WAVE_SORT_PATH, "r", encoding="utf-8") as f:
        path = f.read().strip()
    if not path or not os.path.isdir(path):
        raise FileNotFoundError(f"Invalid Wave_sort path from {LAST_WAVE_SORT_PATH}: {path!r}")
    return path


def make_objective(cli_args):
    def objective(trial: optuna.Trial) -> float:
        tau_info = trial.suggest_categorical("tau_info", list(config.OPTUNA_TAU_INFO_SWEEP))
        weight_info = trial.suggest_categorical("weight_info", list(config.WEIGHT_INFO_SWEEP))

        run_suffix = f"optuna{trial.number:04d}"

        _run_training(
            src=cli_args.src,
            tgt=cli_args.tgt,
            regions=cli_args.regions,
            weight_info=weight_info,
            tau_info=tau_info,
            run_suffix=run_suffix,
            gpu=cli_args.gpu,
            seed=cli_args.seed,
            max_iter=cli_args.max_iter,
        )

        wave_path = _read_wave_sort_path()
        print(f"\nEvaluating trial {trial.number} ...")
        sys.stdout.flush()
        result, _ = run_eval(wave_path, return_details=True, verbose=False)
        hr = result.get("HR")
        if not hr or "MAE" not in hr:
            raise RuntimeError(f"run_eval did not return HR/MAE for {wave_path}")

        mae = float(hr["MAE"])
        trial.set_user_attr("rmse", float(hr.get("RMSE", float("nan"))))
        trial.set_user_attr("me", float(hr.get("ME", float("nan"))))
        trial.set_user_attr("wave_sort_path", wave_path)
        trial.set_user_attr("run_suffix", run_suffix)
        print(f"Trial {trial.number}: MAE={mae:.4f}  "
              f"(loss={config.LOSS_TYPE}, w={weight_info}, tau={tau_info:.4f})")
        return mae

    return objective


def _save_best_result(study: optuna.Study, out_path: str, cli_args) -> None:
    if study.best_trial is None:
        return
    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "src": cli_args.src,
        "tgt": cli_args.tgt,
        "regions": cli_args.regions,
        "loss_type": config.LOSS_TYPE,
        "best_value_mae": best.value,
        "best_params": best.params,
        "best_trial_number": best.number,
        "best_user_attrs": dict(best.user_attrs),
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved best-params JSON: {out_path}")


def main():
    cli_args = parse_args()
    study_name = cli_args.study_name or _default_study_name(
        cli_args.src, cli_args.tgt, cli_args.regions
    )
    storage = cli_args.storage or _default_storage(study_name)

    if cli_args.fresh and storage.startswith("sqlite:///"):
        db_path = storage[len("sqlite:///"):]
        if os.path.isfile(db_path):
            os.remove(db_path)
            print(f"Removed existing study DB: {db_path}")

    grid_size = len(config.OPTUNA_TAU_INFO_SWEEP) * len(config.WEIGHT_INFO_SWEEP)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=_grid_sampler(),
    )

    print("Optuna study:", study_name)
    print("Storage:", storage)
    print("Search space (grid):")
    print("  tau_info    :", list(config.OPTUNA_TAU_INFO_SWEEP))
    print("  weight_info :", list(config.WEIGHT_INFO_SWEEP))
    print("  grid size   :", grid_size, "unique combinations")
    print("  loss_type   :", config.LOSS_TYPE, "(fixed)")
    print("Source -> target:", cli_args.src, "->", cli_args.tgt)
    print("Regions:", cli_args.regions)
    print("Trials:", cli_args.n_trials)
    if cli_args.n_trials > grid_size:
        print(f"Note: --n-trials {cli_args.n_trials} > grid size {grid_size}; "
              f"extra trials will repeat combinations.")

    study.optimize(
        make_objective(cli_args),
        n_trials=cli_args.n_trials,
        timeout=cli_args.timeout,
        show_progress_bar=True,
    )

    if study.best_trial is None:
        print("No completed trials.")
        return

    print("\n" + "=" * 60)
    print("Best trial:", study.best_trial.number)
    print("Best MAE:", study.best_value)
    print("Best params:", study.best_params)
    if study.best_trial.user_attrs.get("wave_sort_path"):
        print("Wave_sort:", study.best_trial.user_attrs["wave_sort_path"])
    print("=" * 60)

    out_path = os.path.join(
        config.RESULT_LOG_DIR,
        "optuna",
        f"{study_name}_best.json",
    )
    _save_best_result(study, out_path, cli_args)


if __name__ == "__main__":
    main()
