from pydantic import BaseModel, Field


class SeedAnalysisRequest(BaseModel):
    experiment_ids: list[str] = Field(min_length=2)


class SeedResult(BaseModel):
    experiment_id: str
    name: str
    seed: int
    test_accuracy: float
    test_loss: float


class SeedAnalysisResponse(BaseModel):
    run_count: int
    mean_test_accuracy: float
    accuracy_standard_deviation: float
    mean_test_loss: float
    loss_standard_deviation: float
    best_experiment_id: str
    worst_experiment_id: str
    configuration_mismatches: dict
    runs: list[SeedResult]