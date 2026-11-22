from pydantic import BaseModel


class HyperparameterSuggestion(BaseModel):
    parameter: str
    current_value: str
    suggested_value: str
    reason: str
    evidence_used: list[str]


class SuggestionResponse(BaseModel):
    experiment_id: str
    suggestions: list[HyperparameterSuggestion]