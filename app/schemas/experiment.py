from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=3, max_length=100)
    dataset_name: Literal["fashion_mnist", "cifar10"] = "fashion_mnist"
    model_name: Literal["simple_cnn"] = "simple_cnn"

    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=32, ge=8, le=512)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    seed: int = Field(default=42, ge=0)
    early_stopping_patience: int = Field(
        default=3,
        ge=1,
        le=20,
    )


class ExperimentStatusUpdate(BaseModel):
    status: ExperimentStatus
    error_message: str | None = None


class ExperimentResponse(BaseModel):
    id: str
    name: str
    status: ExperimentStatus
    dataset_name: str
    model_name: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_message: str | None = None
    config: dict


class EpochMetricResponse(BaseModel):
    epoch: int
    train_loss: float | None = None
    validation_loss: float | None = None
    train_accuracy: float | None = None
    validation_accuracy: float | None = None
    learning_rate: float | None = None
    epoch_time_sec: float | None = None


class EvaluationResponse(BaseModel):
    experiment_id: str
    test_loss: float
    test_accuracy: float
    created_at: datetime


class ClassMetricResponse(BaseModel):
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int


class ConfusedPairResponse(BaseModel):
    actual_class: str
    predicted_class: str
    count: int


class ClassificationEvaluationResponse(BaseModel):
    experiment_id: str
    class_names: list[str]
    confusion_matrix: list[list[int]]
    class_metrics: list[ClassMetricResponse]
    most_confused_pairs: list[ConfusedPairResponse]


class DiagnosticResponse(BaseModel):
    finding_type: str
    severity: str
    title: str
    evidence: dict
    recommendations: list[str]
    rule_version: str