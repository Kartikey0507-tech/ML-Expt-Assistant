import json

from fastapi import HTTPException

from app.storage.database import get_connection


def create_markdown_report(experiment_id: str) -> str:
    """Build one self-contained Markdown report from stored experiment facts."""
    with get_connection() as connection:
        experiment = connection.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()

        evaluation = connection.execute(
            """
            SELECT test_loss, test_accuracy, created_at
            FROM evaluations
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

        classification = connection.execute(
            """
            SELECT class_metrics_json, most_confused_pairs_json
            FROM classification_evaluations
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

        diagnostics = connection.execute(
            """
            SELECT title, severity, evidence_json, recommendations_json
            FROM diagnostics
            WHERE experiment_id = ?
            ORDER BY id
            """,
            (experiment_id,),
        ).fetchall()

        explanation = connection.execute(
            """
            SELECT explanation_json, model, grounding_passed
            FROM explanations
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()

    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    if evaluation is None:
        raise HTTPException(
            status_code=409,
            detail="A completed evaluation is required before creating a report.",
        )

    config = json.loads(experiment["config_json"])

    lines = [
        f"# ML Experiment Report: {experiment['name']}",
        "",
        "## Run summary",
        "",
        f"- **Experiment ID:** `{experiment['id']}`",
        f"- **Status:** {experiment['status']}",
        f"- **Dataset:** {experiment['dataset_name']}",
        f"- **Model:** {experiment['model_name']}",
        f"- **Created:** {experiment['created_at']}",
        f"- **Completed:** {experiment['finished_at'] or 'Not recorded'}",
        "",
        "## Final evaluation",
        "",
        f"- **Test accuracy:** {evaluation['test_accuracy']:.2%}",
        f"- **Test loss:** {evaluation['test_loss']:.4f}",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Reproducibility",
        "",
        f"- **Dataset hash:** `{experiment['dataset_hash'] or 'Not recorded'}`",
        f"- **Git commit:** `{experiment['git_commit'] or 'Not recorded'}`",
        f"- **TensorFlow version:** `{experiment['tf_version'] or 'Not recorded'}`",
        f"- **Artifacts:** `{experiment['artifact_directory'] or 'Not recorded'}`",
        "",
    ]

    if classification is not None:
        class_metrics = json.loads(classification["class_metrics_json"])
        confused_pairs = json.loads(
            classification["most_confused_pairs_json"]
        )

        lines.extend(
            [
                "## Per-class performance",
                "",
                "| Class | Precision | Recall | F1 score | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for metric in class_metrics:
            lines.append(
                f"| {metric['class_name']} | "
                f"{metric['precision']:.3f} | "
                f"{metric['recall']:.3f} | "
                f"{metric['f1_score']:.3f} | "
                f"{metric['support']} |"
            )

        lines.extend(
            [
                "",
                "## Most confused class pairs",
                "",
                "| Actual class | Predicted class | Count |",
                "|---|---|---:|",
            ]
        )

        for pair in confused_pairs:
            lines.append(
                f"| {pair['actual_class']} | "
                f"{pair['predicted_class']} | "
                f"{pair['count']} |"
            )

        lines.append("")

    lines.extend(["## Deterministic diagnostics", ""])

    if not diagnostics:
        lines.append("No significant diagnostic findings were stored.")
        lines.append("")

    for finding in diagnostics:
        evidence = json.loads(finding["evidence_json"])
        recommendations = json.loads(finding["recommendations_json"])

        lines.extend(
            [
                f"### [{finding['severity'].upper()}] {finding['title']}",
                "",
                "**Evidence**",
                "",
                "```json",
                json.dumps(evidence, indent=2),
                "```",
                "",
                "**Recommended actions**",
                "",
            ]
        )

        for recommendation in recommendations:
            lines.append(f"- {recommendation}")

        lines.append("")

    lines.extend(["## Grounded Gemini explanation", ""])

    if explanation is None:
        lines.append(
            "No Gemini explanation was generated for this experiment."
        )
    else:
        generated = json.loads(explanation["explanation_json"])

        lines.extend(
            [
                f"**Model:** `{explanation['model']}`",
                f"**Grounding passed:** `{bool(explanation['grounding_passed'])}`",
                "",
                generated["summary"],
                "",
                "**Recommended actions**",
                "",
            ]
        )

        for action in generated["recommended_actions"]:
            lines.append(f"- **{action['action']}** — {action['rationale']}")

    lines.append("")
    return "\n".join(lines)