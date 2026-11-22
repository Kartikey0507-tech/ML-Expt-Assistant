import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.experiment import ExperimentStatus
from app.services.lifecycle import get_experiment_row, transition_experiment
from app.storage.database import get_connection


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_git_commit() -> str:
    """Return the current source revision, or a clear local-development value."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def save_diagnostics(experiment_id: str, summary: dict) -> None:
    from app.diagnostics.analyzer import RULE_VERSION, analyze_experiment

    with get_connection() as connection:
        metric_rows = connection.execute(
            """
            SELECT epoch, train_loss, validation_loss,
                   train_accuracy, validation_accuracy
            FROM epoch_metrics
            WHERE experiment_id = ?
            ORDER BY epoch
            """,
            (experiment_id,),
        ).fetchall()

    findings = analyze_experiment(
        epoch_metrics=[dict(row) for row in metric_rows],
        class_metrics=summary["class_metrics"],
        most_confused_pairs=summary["most_confused_pairs"],
    )

    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM diagnostics WHERE experiment_id = ?",
            (experiment_id,),
        )

        for finding in findings:
            connection.execute(
                """
                INSERT INTO diagnostics (
                    experiment_id, finding_type, severity, title,
                    evidence_json, recommendations_json,
                    rule_version, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    finding["finding_type"],
                    finding["severity"],
                    finding["title"],
                    json.dumps(finding["evidence"]),
                    json.dumps(finding["recommendations"]),
                    RULE_VERSION,
                    created_at,
                ),
            )

        connection.commit()


def main(experiment_id: str) -> None:
    try:
        row = get_experiment_row(experiment_id)
        config = json.loads(row["config_json"])

        artifact_directory = PROJECT_ROOT / "artifacts" / experiment_id
        artifact_directory.mkdir(parents=True, exist_ok=True)

        with (artifact_directory / "config.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(config, file, indent=2)

        from app.training.trainer import evaluate_and_save, train_model

        model, x_test, y_test, dataset_hash = train_model(
            experiment_id=experiment_id,
            config=config,
            artifact_directory=artifact_directory,
        )

        transition_experiment(experiment_id, ExperimentStatus.EVALUATING)

        summary = evaluate_and_save(
        model=model,
        x_test=x_test,
        y_test=y_test,
        class_names=class_names,
        artifact_directory=artifact_directory,
        )

        completed_at = datetime.now(timezone.utc).isoformat()
        git_commit = get_git_commit()

        environment = {
            "experiment_id": experiment_id,
            "dataset_name": row["dataset_name"],
            "dataset_hash": dataset_hash,
            "git_commit": git_commit,
            "python_version": sys.version,
            "tensorflow_version": summary["tensorflow_version"],
            "completed_at": completed_at,
        }

        with (artifact_directory / "environment.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(environment, file, indent=2)

        with get_connection() as connection:
            connection.execute(
                """
                UPDATE experiments
                SET dataset_hash = ?,
                    git_commit = ?,
                    tf_version = ?,
                    artifact_directory = ?
                WHERE id = ?
                """,
                (
                    dataset_hash,
                    git_commit,
                    summary["tensorflow_version"],
                    str(artifact_directory),
                    experiment_id,
                ),
            )

            connection.execute(
                """
                INSERT OR REPLACE INTO evaluations (
                    experiment_id, test_loss, test_accuracy, created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    summary["test_loss"],
                    summary["test_accuracy"],
                    completed_at,
                ),
            )

            connection.execute(
                """
                INSERT OR REPLACE INTO classification_evaluations (
                    experiment_id, class_names_json, confusion_matrix_json,
                    class_metrics_json, most_confused_pairs_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    json.dumps(summary["class_names"]),
                    json.dumps(summary["confusion_matrix"]),
                    json.dumps(summary["class_metrics"]),
                    json.dumps(summary["most_confused_pairs"]),
                    completed_at,
                ),
            )
            connection.commit()

        save_diagnostics(experiment_id, summary)
        transition_experiment(experiment_id, ExperimentStatus.COMPLETED)

    except Exception as error:
        error_details = f"{type(error).__name__}: {error}"

        try:
            transition_experiment(
                experiment_id,
                ExperimentStatus.FAILED,
                error_message=error_details,
            )
        except Exception:
            traceback.print_exc()

        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    arguments = parser.parse_args()
    main(arguments.experiment_id)