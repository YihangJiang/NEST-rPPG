import os
import mlflow
from mlflow.tracking import MlflowClient

import config

tracking_uri = config.MLFLOW_TRACKING_URI
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri)

print("Tracking URI:", tracking_uri)

experiments = client.search_experiments()
deleted_count = 0

for exp in experiments:
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.run_name LIKE '%optuna%'",
        max_results=5000,
    )
    for run in runs:
        print(f"Deleting MLflow run: {run.info.run_id} | name: {run.info.run_name}")
        client.delete_run(run.info.run_id)
        deleted_count += 1

print(f"\nSuccessfully deleted {deleted_count} optuna run(s) from MLflow database.")
