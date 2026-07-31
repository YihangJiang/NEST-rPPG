"""Optional MLflow experiment tracking for NEST-rPPG training and eval."""
import os
from typing import Any, Dict, Optional, Union

import config

_mlflow = None
_active = False
_run_id: Optional[str] = None

LAST_RUN_ID_FILE = os.path.join(config.RESULT_LOG_DIR, "last_mlflow_run_id.txt")
DEFAULT_MODEL_ARTIFACT_PATH = "model"


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
    try:
        mlflow = _import_mlflow()
    except ImportError as exc:
        print(f"[MLflow] {exc}")
        return False

    os.makedirs(config.RESULT_LOG_DIR, exist_ok=True)
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


def _model_pip_requirements() -> list[str]:
    """Pinned requirements using exact installed versions (keeps CUDA local labels)."""
    import importlib.metadata as metadata

    packages = ("torch", "torchvision", "cloudpickle", "mlflow")
    reqs: list[str] = []
    for package in packages:
        try:
            reqs.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            continue
    return reqs


def log_model(
    model,
    artifact_path: str = DEFAULT_MODEL_ARTIFACT_PATH,
    registered_model_name: Optional[str] = None,
) -> None:
    if not _active:
        return
    import mlflow.pytorch

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name=artifact_path,
        registered_model_name=registered_model_name,
        serialization_format="pickle",
        pip_requirements=_model_pip_requirements(),
    )


def get_run_id() -> Optional[str]:
    return _run_id


def _find_run_id_by_name(
    run_name: str,
    experiment_name: Optional[str] = None,
) -> Optional[str]:
    from mlflow import MlflowClient

    client = MlflowClient(_tracking_uri())
    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_ids = [experiment.experiment_id] if experiment else []
    else:
        experiment_ids = [exp.experiment_id for exp in client.search_experiments()]

    if not experiment_ids:
        return None

    safe_name = run_name.replace('"', '\\"')
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=f'attributes.run_name = "{safe_name}"',
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return runs[0].info.run_id


def load_model(
    *,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    artifact_path: str = DEFAULT_MODEL_ARTIFACT_PATH,
    map_location=None,
):
    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(_tracking_uri())

    resolved_run_id = run_id or _load_run_id()
    if resolved_run_id is None and run_name:
        resolved_run_id = _find_run_id_by_name(run_name, experiment_name)
    if not resolved_run_id:
        raise FileNotFoundError(
            "No MLflow run found for model load. "
            "Train with MLflow first or pass run_id/run_name."
        )

    import mlflow.pytorch

    model_uri = f"runs:/{resolved_run_id}/{artifact_path}"
    return mlflow.pytorch.load_model(model_uri, map_location=map_location)


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
