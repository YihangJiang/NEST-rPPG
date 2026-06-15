"""Optional MLflow experiment tracking for NEST-rPPG training and eval."""
import os
from typing import Any, Dict, Optional, Union

import config

_mlflow = None
_active = False
_run_id: Optional[str] = None

LAST_RUN_ID_FILE = os.path.join(config.RESULT_LOG_DIR, "last_mlflow_run_id.txt")


def _import_mlflow():
    global _mlflow
    if _mlflow is None:
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow is not installed. Install with: pip install mlflow"
            ) from exc
        _mlflow = mlflow
    return _mlflow


def is_enabled(args=None) -> bool:
    if args is not None and getattr(args, "mlflow", False):
        return True
    return os.environ.get("MLFLOW_ENABLED", "").lower() in ("1", "true", "yes")


def _tracking_uri(args=None) -> str:
    if args is not None and getattr(args, "mlflow_tracking_uri", None):
        return args.mlflow_tracking_uri
    return os.environ.get("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI)


def _experiment_name(args=None, default: Optional[str] = None) -> str:
    if args is not None and getattr(args, "mlflow_experiment", None):
        return args.mlflow_experiment
    return os.environ.get("MLFLOW_EXPERIMENT_NAME", default or config.MLFLOW_EXPERIMENT_NAME)


def setup(
    args=None,
    *,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> bool:
    """Start an MLflow run. Returns True when tracking is active."""
    global _active, _run_id
    if not is_enabled(args):
        return False

    try:
        mlflow = _import_mlflow()
    except ImportError as exc:
        print(f"[MLflow] {exc}")
        return False

    os.makedirs(config.MLFLOW_ARTIFACT_ROOT, exist_ok=True)
    mlflow.set_tracking_uri(_tracking_uri(args))
    mlflow.set_experiment(experiment_name or _experiment_name(args))

    if args is not None and getattr(args, "mlflow_run_name", None):
        run_name = args.mlflow_run_name

    mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)

    _active = True
    _run_id = mlflow.active_run().info.run_id
    _save_run_id(_run_id)
    return True


def log_params(params: Dict[str, Any]) -> None:
    if not _active:
        return
    safe = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in params.items()}
    _import_mlflow().log_params(safe)


def log_metrics(metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
    if not _active:
        return
    _import_mlflow().log_metrics(
        {k: float(v) for k, v in metrics.items()},
        step=step,
    )


def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
    if not _active or not path or not os.path.exists(path):
        return
    _import_mlflow().log_artifact(path, artifact_path)


def log_artifacts(paths) -> None:
    for path in paths:
        log_artifact(path)


def end_run() -> None:
    global _active, _run_id
    if not _active:
        return
    _import_mlflow().end_run()
    _active = False
    _run_id = None


def resume_run(run_id: Optional[str] = None) -> bool:
    """Resume logging to an existing run (e.g. eval after training)."""
    global _active, _run_id
    if not is_enabled():
        return False

    run_id = run_id or _load_run_id()
    if not run_id:
        return False

    try:
        mlflow = _import_mlflow()
    except ImportError as exc:
        print(f"[MLflow] {exc}")
        return False

    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.start_run(run_id=run_id)
    _active = True
    _run_id = run_id
    return True


def _save_run_id(run_id: str) -> None:
    os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
    with open(LAST_RUN_ID_FILE, "w") as f:
        f.write(run_id + "\n")


def _load_run_id() -> Optional[str]:
    if not os.path.isfile(LAST_RUN_ID_FILE):
        return None
    with open(LAST_RUN_ID_FILE) as f:
        return f.read().strip() or None
