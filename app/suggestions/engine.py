import json

from app.schemas.suggestions import (
    HyperparameterSuggestion,
    SuggestionResponse,
)
from app.services.lifecycle import get_experiment_row
from app.storage.database import get_connection


def generate_suggestions(experiment_id: str) -> SuggestionResponse:
    """Generate safe, deterministic next-run recommendations."""
    experiment = get_experiment_row(experiment_id)
    config = json.loads(experiment["config_json"])

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT finding_type, evidence_json
            FROM diagnostics
            WHERE experiment_id = ?
            ORDER BY id
            """,
            (experiment_id,),
        ).fetchall()

    suggestions = []

    for row in rows:
        finding_type = row["finding_type"]
        evidence = json.loads(row["evidence_json"])

        if finding_type == "overfitting":
            best_epoch = evidence["best_validation_epoch"]

            suggestions.append(
                HyperparameterSuggestion(
                    parameter="epochs",
                    current_value=str(config["epochs"]),
                    suggested_value=str(best_epoch),
                    reason=(
                        "Validation loss was best at the suggested epoch, "
                        "then became worse while training loss improved."
                    ),
                    evidence_used=[
                        "best_validation_epoch",
                        "best_validation_loss",
                        "final_loss_gap",
                    ],
                )
            )

            suggestions.append(
                HyperparameterSuggestion(
                    parameter="early_stopping_patience",
                    current_value=str(
                        config.get("early_stopping_patience", 3)
                    ),
                    suggested_value="2",
                    reason=(
                        "Stopping sooner can reduce training after "
                        "validation performance begins worsening."
                    ),
                    evidence_used=[
                        "best_validation_epoch",
                        "final_validation_loss",
                    ],
                )
            )

        elif finding_type == "underfitting":
            current_epochs = config["epochs"]

            suggestions.append(
                HyperparameterSuggestion(
                    parameter="epochs",
                    current_value=str(current_epochs),
                    suggested_value=str(current_epochs + 10),
                    reason=(
                        "Training accuracy remained low, indicating the "
                        "model may need more learning time."
                    ),
                    evidence_used=[
                        "final_training_accuracy",
                        "epochs_completed",
                    ],
                )
            )

        elif finding_type == "plateau":
            current_learning_rate = config["learning_rate"]
            suggested_learning_rate = current_learning_rate / 2

            suggestions.append(
                HyperparameterSuggestion(
                    parameter="learning_rate",
                    current_value=str(current_learning_rate),
                    suggested_value=f"{suggested_learning_rate:.6f}",
                    reason=(
                        "Validation accuracy changed very little in recent "
                        "epochs; a smaller learning rate may refine learning."
                    ),
                    evidence_used=[
                        "recent_validation_accuracy_range",
                        "final_validation_accuracy",
                    ],
                )
            )

        elif finding_type == "weak_class_recall":
            suggestions.append(
                HyperparameterSuggestion(
                    parameter="data_augmentation",
                    current_value="not configured",
                    suggested_value="add targeted augmentation",
                    reason=(
                        "The identified class has weak recall. Targeted "
                        "augmentation can improve robustness for that class."
                    ),
                    evidence_used=[
                        "class_name",
                        "recall",
                        "f1_score",
                    ],
                )
            )

        elif finding_type == "class_confusion":
            suggestions.append(
                HyperparameterSuggestion(
                    parameter="model_architecture",
                    current_value=config["model_name"],
                    suggested_value="deeper_cnn",
                    reason=(
                        "The model frequently confuses two visual classes; "
                        "more feature-extraction capacity may help."
                    ),
                    evidence_used=[
                        "actual_class",
                        "predicted_class",
                        "confusion_count",
                    ],
                )
            )

    if not suggestions:
        suggestions.append(
            HyperparameterSuggestion(
                parameter="next_experiment",
                current_value="current configuration",
                suggested_value="change one setting at a time",
                reason=(
                    "No major deterministic issue was found. Controlled "
                    "one-variable experiments are the safest next step."
                ),
                evidence_used=[],
            )
        )

    return SuggestionResponse(
        experiment_id=experiment_id,
        suggestions=suggestions,
    )