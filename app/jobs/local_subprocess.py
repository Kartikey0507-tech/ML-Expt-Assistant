import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def start_training_job(experiment_id: str) -> int:
    """Start a training worker without blocking the FastAPI server."""
    artifact_directory = PROJECT_ROOT / "artifacts" / experiment_id
    artifact_directory.mkdir(parents=True, exist_ok=True)

    log_path = artifact_directory / "training.log"

    command = [
        sys.executable,
        "-m",
        "app.jobs.training_job",
        "--experiment-id",
        experiment_id,
    ]

    creation_flags = 0
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NO_WINDOW

    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )

    return process.pid