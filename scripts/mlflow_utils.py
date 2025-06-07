import os

import hydra
import mlflow
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def setup_mlflow(cfg: DictConfig):
    """Configure MLflow tracking"""
    url = (cfg.ml_flow.url,)
    mlflow.set_tracking_uri(url)
    mlflow.set_experiment("NQ_model")
    mlflow.start_run()

    mlflow.log_params(
        {
            "platform": os.uname().sysname,
            "python_version": os.sys.version,
            "cpu_count": os.cpu_count(),
        }
    )


class MLflowLogger:
    def __init__(self):
        self.current_epoch = 0

    def log_metrics(self, metrics: dict):
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics, step=self.current_epoch)

    def log_params(self, params: dict):
        """Log parameters to MLflow"""
        mlflow.log_params(params)

    def log_model(self, model, artifact_path: str):
        """Log model artifacts"""
        mlflow.pytorch.log_model(model, artifact_path)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
