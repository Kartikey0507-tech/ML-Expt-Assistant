from datetime import datetime, timezone

from fastapi import HTTPException, status

from app.schemas.experiment import ExperimentStatus
from app.storage.database import get_connection


ALLOWED_TRANSITIONS: dict[ExperimentStatus, set[ExperimentStatus]] = {
    ExperimentStatus.QUEUED: {
        ExperimentStatus.RUNNING,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.RUNNING: {
        ExperimentStatus.EVALUATING,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.EVALUATING: {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.COMPLETED: set(),
    ExperimentStatus.FAILED: set(),
}


def validate_status_transition(
    current_status: ExperimentStatus,
    next_status: ExperimentStatus,
) -> None:
    """Reject invalid experiment state changes."""
    if next_status not in ALLOWED_TRANSITIONS[current_status]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot change experiment from '{current_status.value}' "
                f"to '{next_status.value}'."
            ),
        )


def get_experiment_row(experiment_id: str):
    """Fetch an experiment database row or return a 404 error."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    return row


def transition_experiment(
    experiment_id: str,
    next_status: ExperimentStatus,
    error_message: str | None = None,
):
    """Safely move an experiment to its next allowed state."""
    row = get_experiment_row(experiment_id)
    current_status = ExperimentStatus(row["status"])

    validate_status_transition(current_status, next_status)

    now = datetime.now(timezone.utc).isoformat()
    started_at = row["started_at"]
    finished_at = row["finished_at"]

    if next_status == ExperimentStatus.RUNNING and not started_at:
        started_at = now

    if next_status in {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED}:
        finished_at = now

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE experiments
            SET status = ?, started_at = ?, finished_at = ?, error_message = ?
            WHERE id = ?
            """,
            (
                next_status.value,
                started_at,
                finished_at,
                error_message if error_message else row["error_message"],
                experiment_id,
            ),
        )
        connection.commit()

    return get_experiment_row(experiment_id)