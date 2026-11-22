import json

from fastapi import APIRouter, HTTPException

from app.schemas.comparison import (
    ComparedExperiment,
    ComparisonResponse,
)
from app.storage.database import get_connection


router = APIRouter(prefix="/comparisons", tags=["comparisons"])


def load_completed_experiment(experiment_id: str) -> ComparedExperiment:
    """Load one completed experiment and the facts required for comparison."""
    with get_connection() as connection:
        experiment = connection.execute(
            """
            SELECT id, name, status, config_json
            FROM experiments
            WHERE id = ?
            """,
            (experiment_id,),
        ).fetchone()

        evaluation = connection.execute(
            """
            SELECT test_accuracy, test_loss
            FROM evaluations
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

        diagnostic_rows = connection.execute(
            """
            SELECT title
            FROM diagnostics
            WHERE experiment_id = ?
            ORDER BY id
            """,
            (experiment_id,),
        ).fetchall()

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    if experiment["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Only completed experiments can be compared.",
        )

    if evaluation is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This experiment has no saved final evaluation. "
                "Use a newer completed experiment."
            ),
        )

    return ComparedExperiment(
        id=experiment["id"],
        name=experiment["name"],
        test_accuracy=evaluation["test_accuracy"],
        test_loss=evaluation["test_loss"],
        config=json.loads(experiment["config_json"]),
        diagnostics=[row["title"] for row in diagnostic_rows],
    )


@router.get("/", response_model=ComparisonResponse)
def compare_experiments(
    left_experiment_id: str,
    right_experiment_id: str,
) -> ComparisonResponse:
    """Compare two completed experiments using stored evaluation results."""
    if left_experiment_id == right_experiment_id:
        raise HTTPException(
            status_code=400,
            detail="Choose two different experiments to compare.",
        )

    left = load_completed_experiment(left_experiment_id)
    right = load_completed_experiment(right_experiment_id)

    accuracy_difference = right.test_accuracy - left.test_accuracy
    loss_difference = right.test_loss - left.test_loss

    configuration_differences = {
        key: {
            "left": left.config.get(key),
            "right": right.config.get(key),
        }
        for key in sorted(set(left.config) | set(right.config))
        if left.config.get(key) != right.config.get(key)
    }

    if right.test_accuracy > left.test_accuracy:
        winner_id = right.id
        winner_reason = (
            f"'{right.name}' has higher test accuracy "
            f"({right.test_accuracy:.2%} vs {left.test_accuracy:.2%})."
        )
    elif left.test_accuracy > right.test_accuracy:
        winner_id = left.id
        winner_reason = (
            f"'{left.name}' has higher test accuracy "
            f"({left.test_accuracy:.2%} vs {right.test_accuracy:.2%})."
        )
    elif right.test_loss < left.test_loss:
        winner_id = right.id
        winner_reason = (
            f"Both runs have equal test accuracy, but '{right.name}' "
            "has lower test loss."
        )
    else:
        winner_id = left.id
        winner_reason = (
            f"Both runs have equal test accuracy, but '{left.name}' "
            "has lower or equal test loss."
        )

    return ComparisonResponse(
        left=left,
        right=right,
        accuracy_difference=accuracy_difference,
        loss_difference=loss_difference,
        winner_id=winner_id,
        winner_reason=winner_reason,
        configuration_differences=configuration_differences,
    )