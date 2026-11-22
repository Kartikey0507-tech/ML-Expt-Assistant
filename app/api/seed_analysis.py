import json
import statistics

from fastapi import APIRouter, HTTPException

from app.schemas.seed_analysis import (
    SeedAnalysisRequest,
    SeedAnalysisResponse,
    SeedResult,
)
from app.storage.database import get_connection


router = APIRouter(prefix="/seed-analysis", tags=["seed analysis"])


@router.post("/", response_model=SeedAnalysisResponse)
def analyze_seeds(request: SeedAnalysisRequest) -> SeedAnalysisResponse:
    """Calculate statistics across completed runs with different seeds."""
    runs = []

    with get_connection() as connection:
        for experiment_id in request.experiment_ids:
            row = connection.execute(
                """
                SELECT
                    experiments.id,
                    experiments.name,
                    experiments.status,
                    experiments.config_json,
                    evaluations.test_accuracy,
                    evaluations.test_loss
                FROM experiments
                LEFT JOIN evaluations
                    ON evaluations.experiment_id = experiments.id
                WHERE experiments.id = ?
                """,
                (experiment_id,),
            ).fetchone()

            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment '{experiment_id}' was not found.",
                )

            if row["status"] != "completed" or row["test_accuracy"] is None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Experiment '{row['name']}' needs a completed "
                        "evaluation before seed analysis."
                    ),
                )

            config = json.loads(row["config_json"])

            runs.append(
                {
                    "experiment_id": row["id"],
                    "name": row["name"],
                    "seed": config["seed"],
                    "test_accuracy": row["test_accuracy"],
                    "test_loss": row["test_loss"],
                    "config": config,
                }
            )

    baseline_config = {
        key: value
        for key, value in runs[0]["config"].items()
        if key != "seed"
    }

    configuration_mismatches = {}

    for run in runs[1:]:
        comparable_config = {
            key: value
            for key, value in run["config"].items()
            if key != "seed"
        }

        if comparable_config != baseline_config:
            configuration_mismatches[run["experiment_id"]] = {
                "name": run["name"],
                "config": comparable_config,
            }

    accuracies = [run["test_accuracy"] for run in runs]
    losses = [run["test_loss"] for run in runs]

    best_run = max(runs, key=lambda run: run["test_accuracy"])
    worst_run = min(runs, key=lambda run: run["test_accuracy"])

    return SeedAnalysisResponse(
        run_count=len(runs),
        mean_test_accuracy=statistics.mean(accuracies),
        accuracy_standard_deviation=statistics.stdev(accuracies),
        mean_test_loss=statistics.mean(losses),
        loss_standard_deviation=statistics.stdev(losses),
        best_experiment_id=best_run["experiment_id"],
        worst_experiment_id=worst_run["experiment_id"],
        configuration_mismatches=configuration_mismatches,
        runs=[
            SeedResult(
                experiment_id=run["experiment_id"],
                name=run["name"],
                seed=run["seed"],
                test_accuracy=run["test_accuracy"],
                test_loss=run["test_loss"],
            )
            for run in runs
        ],
    )