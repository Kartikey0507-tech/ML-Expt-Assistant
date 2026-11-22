from pydantic import BaseModel


class ComparedExperiment(BaseModel):
    id: str
    name: str
    test_accuracy: float
    test_loss: float
    config: dict
    diagnostics: list[str]


class ComparisonResponse(BaseModel):
    left: ComparedExperiment
    right: ComparedExperiment
    accuracy_difference: float
    loss_difference: float
    winner_id: str
    winner_reason: str
    configuration_differences: dict