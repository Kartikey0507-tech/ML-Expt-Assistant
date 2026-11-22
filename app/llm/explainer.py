import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Literal

from dotenv import load_dotenv
from fastapi import HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.services.lifecycle import get_experiment_row
from app.storage.database import get_connection


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

PROMPT_VERSION = "1.0.0"
DEFAULT_MODEL = "gemini-flash-latest"


class RecommendedAction(BaseModel):
    action: str = Field(
        description="One practical next experiment or model change."
    )
    rationale: str = Field(
        description="Reason based only on supplied evidence."
    )
    evidence_used: list[str] = Field(
        description="Exact evidence IDs supporting this action."
    )


class GroundedExplanation(BaseModel):
    summary: str
    severity: Literal["low", "medium", "high"]
    evidence_used: list[str]
    recommended_actions: list[RecommendedAction]
    limitations: list[str]


class ExplanationRecord(GroundedExplanation):
    experiment_id: str
    provider: str
    model: str
    grounding_passed: bool
    generated_at: datetime


def load_evidence_package(experiment_id: str) -> tuple[dict, set[str]]:
    """Load only trusted, stored facts for the Gemini prompt."""
    experiment = get_experiment_row(experiment_id)

    with get_connection() as connection:
        evaluation_row = connection.execute(
            """
            SELECT test_loss, test_accuracy
            FROM evaluations
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

        diagnostic_rows = connection.execute(
            """
            SELECT finding_type, severity, title, evidence_json
            FROM diagnostics
            WHERE experiment_id = ?
            ORDER BY id
            """,
            (experiment_id,),
        ).fetchall()

    if evaluation_row is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation data is required before generating an explanation.",
        )

    findings = []
    evidence_ids = set()

    for index, row in enumerate(diagnostic_rows):
        finding_id = f"finding_{index}"

        evidence = json.loads(row["evidence_json"])

        for evidence_key in evidence:
            evidence_ids.add(f"{finding_id}.{evidence_key}")

        findings.append(
            {
                "id": finding_id,
                "type": row["finding_type"],
                "severity": row["severity"],
                "title": row["title"],
                "evidence": evidence,
            }
        )

    evidence_ids.add("evaluation.test_loss")
    evidence_ids.add("evaluation.test_accuracy")

    package = {
        "experiment_config": json.loads(experiment["config_json"]),
        "evaluation": {
            "test_loss": evaluation_row["test_loss"],
            "test_accuracy": evaluation_row["test_accuracy"],
        },
        "diagnostic_findings": findings,
        "available_evidence_ids": sorted(evidence_ids),
    }

    return package, evidence_ids


def collect_numeric_values(value) -> list[float]:
    """Find all numeric facts from the evidence package."""
    numbers = []

    if isinstance(value, dict):
        for nested_value in value.values():
            numbers.extend(collect_numeric_values(nested_value))

    elif isinstance(value, list):
        for nested_value in value:
            numbers.extend(collect_numeric_values(nested_value))

    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numbers.append(float(value))

    return numbers


def verify_grounding(
    explanation: GroundedExplanation,
    evidence_package: dict,
    allowed_evidence_ids: set[str],
) -> None:
    """Reject explanations that cite unavailable evidence or invented numbers."""
    cited_ids = set(explanation.evidence_used)

    for action in explanation.recommended_actions:
        cited_ids.update(action.evidence_used)

    unknown_ids = cited_ids - allowed_evidence_ids

    if unknown_ids:
        raise ValueError(
            f"Explanation cited unknown evidence IDs: {sorted(unknown_ids)}"
        )

    allowed_numbers = collect_numeric_values(evidence_package)

    explanation_text = explanation.summary
    for action in explanation.recommended_actions:
        explanation_text += f" {action.action} {action.rationale}"

    number_strings = re.findall(r"\b\d+(?:\.\d+)?\b", explanation_text)

    for number_string in number_strings:
        number = float(number_string)

        if not any(
            math.isclose(number, allowed, rel_tol=1e-6, abs_tol=1e-6)
            for allowed in allowed_numbers
        ):
            raise ValueError(
                f"Explanation contains unsupported number: {number_string}"
            )


def generate_explanation(experiment_id: str) -> ExplanationRecord:
    """Call Gemini once, validate output, and save the grounded explanation."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the root .env file."
        )

    evidence_package, allowed_evidence_ids = load_evidence_package(experiment_id)

    prompt = f"""
You are an ML experiment assistant.

Explain the model result using ONLY the evidence JSON below.

Rules:
- Do not diagnose anything not present in diagnostic_findings.
- Do not invent numerical values.
- Use numeric values exactly as supplied; do not convert decimals to percentages.
- Every explanation and action must cite one or more IDs from available_evidence_ids.
- Prefer 1 to 3 concrete next actions.
- State limitations when there is insufficient evidence.

Evidence JSON:
{json.dumps(evidence_package, indent=2)}
"""

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    client = genai.Client(api_key=api_key)

    started_at = time.perf_counter()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=GroundedExplanation,
        ),
    )

    latency_ms = int((time.perf_counter() - started_at) * 1000)

    explanation = GroundedExplanation.model_validate_json(response.text)

    verify_grounding(
        explanation=explanation,
        evidence_package=evidence_package,
        allowed_evidence_ids=allowed_evidence_ids,
    )

    evidence_hash = hashlib.sha256(
        json.dumps(evidence_package, sort_keys=True).encode("utf-8")
    ).hexdigest()

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)

    generated_at = datetime.now(timezone.utc)

    with get_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO explanations (
                experiment_id,
                provider,
                model,
                explanation_json,
                evidence_hash,
                grounding_passed,
                prompt_version,
                input_tokens,
                output_tokens,
                latency_ms,
                generated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                "gemini",
                model_name,
                explanation.model_dump_json(),
                evidence_hash,
                1,
                PROMPT_VERSION,
                input_tokens,
                output_tokens,
                latency_ms,
                generated_at.isoformat(),
            ),
        )
        connection.commit()

    return ExplanationRecord(
        experiment_id=experiment_id,
        provider="gemini",
        model=model_name,
        grounding_passed=True,
        generated_at=generated_at,
        **explanation.model_dump(),
    )


def get_saved_explanation(experiment_id: str) -> ExplanationRecord:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                provider,
                model,
                explanation_json,
                grounding_passed,
                generated_at
            FROM explanations
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No explanation has been generated for this experiment.",
        )

    explanation = GroundedExplanation.model_validate_json(
        row["explanation_json"]
    )

    return ExplanationRecord(
        experiment_id=experiment_id,
        provider=row["provider"],
        model=row["model"],
        grounding_passed=bool(row["grounding_passed"]),
        generated_at=datetime.fromisoformat(row["generated_at"]),
        **explanation.model_dump(),
    )