import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.jobs.local_subprocess import start_training_job
from app.schemas.experiment import (
    EpochMetricResponse,
    EvaluationResponse,
    ExperimentCreate,
    ExperimentResponse,
    ExperimentStatus,
    ExperimentStatusUpdate,
    ClassificationEvaluationResponse,
    ClassMetricResponse,
    ConfusedPairResponse,
    DiagnosticResponse,
)
from app.llm.explainer import (
    ExplanationRecord,
    generate_explanation,
    get_saved_explanation,
)
from app.services.lifecycle import get_experiment_row, transition_experiment
from app.storage.database import get_connection

from fastapi.responses import PlainTextResponse

from app.reports.markdown import create_markdown_report

from app.schemas.suggestions import SuggestionResponse
from app.suggestions.engine import generate_suggestions


router = APIRouter(prefix="/experiments", tags=["experiments"])


def row_to_response(row) -> ExperimentResponse:
    return ExperimentResponse(
        id=row["id"],
        name=row["name"],
        status=ExperimentStatus(row["status"]),
        dataset_name=row["dataset_name"],
        model_name=row["model_name"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=(
            datetime.fromisoformat(row["started_at"])
            if row["started_at"]
            else None
        ),
        finished_at=(
            datetime.fromisoformat(row["finished_at"])
            if row["finished_at"]
            else None
        ),
        error_message=row["error_message"],
        config=json.loads(row["config_json"]),
    )


@router.post(
    "/",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment(experiment: ExperimentCreate) -> ExperimentResponse:
    experiment_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO experiments (
                id, name, status, dataset_name, model_name,
                config_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                experiment.name,
                ExperimentStatus.QUEUED.value,
                experiment.dataset_name,
                experiment.model_name,
                experiment.model_dump_json(),
                created_at.isoformat(),
            ),
        )
        connection.commit()

    return row_to_response(get_experiment_row(experiment_id))


@router.get("/", response_model=list[ExperimentResponse])
def list_experiments() -> list[ExperimentResponse]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()

    return [row_to_response(row) for row in rows]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: str) -> ExperimentResponse:
    return row_to_response(get_experiment_row(experiment_id))


@router.get(
    "/{experiment_id}/metrics",
    response_model=list[EpochMetricResponse],
)
def get_epoch_metrics(experiment_id: str) -> list[EpochMetricResponse]:
    get_experiment_row(experiment_id)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                epoch,
                train_loss,
                validation_loss,
                train_accuracy,
                validation_accuracy,
                learning_rate,
                epoch_time_sec
            FROM epoch_metrics
            WHERE experiment_id = ?
            ORDER BY epoch
            """,
            (experiment_id,),
        ).fetchall()

    return [EpochMetricResponse(**dict(row)) for row in rows]


@router.get(
    "/{experiment_id}/evaluation",
    response_model=EvaluationResponse,
)
def get_evaluation(experiment_id: str) -> EvaluationResponse:
    get_experiment_row(experiment_id)

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT experiment_id, test_loss, test_accuracy, created_at
            FROM evaluations
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation is not available until training completes.",
        )

    return EvaluationResponse(
        experiment_id=row["experiment_id"],
        test_loss=row["test_loss"],
        test_accuracy=row["test_accuracy"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


@router.patch("/{experiment_id}/status", response_model=ExperimentResponse)
def update_experiment_status(
    experiment_id: str,
    update: ExperimentStatusUpdate,
) -> ExperimentResponse:
    row = transition_experiment(
        experiment_id=experiment_id,
        next_status=update.status,
        error_message=update.error_message,
    )
    return row_to_response(row)


@router.post("/{experiment_id}/start", response_model=ExperimentResponse)
def start_experiment(experiment_id: str) -> ExperimentResponse:
    row = get_experiment_row(experiment_id)

    if ExperimentStatus(row["status"]) != ExperimentStatus.QUEUED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only queued experiments can be started.",
        )

    transition_experiment(experiment_id, ExperimentStatus.RUNNING)

    try:
        start_training_job(experiment_id)
    except Exception as error:
        row = transition_experiment(
            experiment_id,
            ExperimentStatus.FAILED,
            error_message=f"Could not start worker: {error}",
        )
        return row_to_response(row)

    return row_to_response(get_experiment_row(experiment_id))

@router.get(
    "/{experiment_id}/classification",
    response_model=ClassificationEvaluationResponse,
)
def get_classification_evaluation(
    experiment_id: str,
) -> ClassificationEvaluationResponse:
    get_experiment_row(experiment_id)

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                class_names_json,
                confusion_matrix_json,
                class_metrics_json,
                most_confused_pairs_json
            FROM classification_evaluations
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Classification evaluation is not available yet.",
        )

    return ClassificationEvaluationResponse(
        experiment_id=experiment_id,
        class_names=json.loads(row["class_names_json"]),
        confusion_matrix=json.loads(row["confusion_matrix_json"]),
        class_metrics=[
            ClassMetricResponse(**metric)
            for metric in json.loads(row["class_metrics_json"])
        ],
        most_confused_pairs=[
            ConfusedPairResponse(**pair)
            for pair in json.loads(row["most_confused_pairs_json"])
        ],
    )

@router.get(
    "/{experiment_id}/diagnostics",
    response_model=list[DiagnosticResponse],
)
def get_diagnostics(experiment_id: str) -> list[DiagnosticResponse]:
    get_experiment_row(experiment_id)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                finding_type,
                severity,
                title,
                evidence_json,
                recommendations_json,
                rule_version
            FROM diagnostics
            WHERE experiment_id = ?
            ORDER BY
                CASE severity
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    ELSE 3
                END
            """,
            (experiment_id,),
        ).fetchall()

    return [
        DiagnosticResponse(
            finding_type=row["finding_type"],
            severity=row["severity"],
            title=row["title"],
            evidence=json.loads(row["evidence_json"]),
            recommendations=json.loads(row["recommendations_json"]),
            rule_version=row["rule_version"],
        )
        for row in rows
    ]

@router.post(
    "/{experiment_id}/explanation",
    response_model=ExplanationRecord,
)
def create_explanation(experiment_id: str) -> ExplanationRecord:
    experiment = get_experiment_row(experiment_id)

    if experiment["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="An explanation can be generated only for completed runs.",
        )

    try:
        return generate_explanation(experiment_id)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=f"Grounding validation rejected the response: {error}",
        )


@router.get(
    "/{experiment_id}/explanation",
    response_model=ExplanationRecord,
)
def get_explanation(experiment_id: str) -> ExplanationRecord:
    return get_saved_explanation(experiment_id)

@router.get(
    "/{experiment_id}/report.md",
    response_class=PlainTextResponse,
)
def download_markdown_report(experiment_id: str) -> PlainTextResponse:
    """Return a portable Markdown report for a completed experiment."""
    report = create_markdown_report(experiment_id)

    return PlainTextResponse(
        content=report,
        media_type="text/markdown",
        headers={
            "Content-Disposition": (
                f'attachment; filename="experiment-{experiment_id}.md"'
            )
        },
    )

@router.get(
    "/{experiment_id}/suggestions",
    response_model=SuggestionResponse,
)
def get_hyperparameter_suggestions(
    experiment_id: str,
) -> SuggestionResponse:
    experiment = get_experiment_row(experiment_id)

    if experiment["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Suggestions are available only for completed runs.",
        )

    return generate_suggestions(experiment_id)