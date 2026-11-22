RULE_VERSION = "1.0.0"


def make_finding(
    finding_type: str,
    severity: str,
    title: str,
    evidence: dict,
    recommendations: list[str],
) -> dict:
    return {
        "finding_type": finding_type,
        "severity": severity,
        "title": title,
        "evidence": evidence,
        "recommendations": recommendations,
    }


def analyze_experiment(
    epoch_metrics: list[dict],
    class_metrics: list[dict],
    most_confused_pairs: list[dict],
) -> list[dict]:
    """Return rule-based, explainable findings from stored experiment facts."""
    findings = []

    if len(epoch_metrics) >= 3:
        train_losses = [row["train_loss"] for row in epoch_metrics]
        validation_losses = [row["validation_loss"] for row in epoch_metrics]
        train_accuracies = [row["train_accuracy"] for row in epoch_metrics]
        validation_accuracies = [row["validation_accuracy"] for row in epoch_metrics]

        best_validation_loss = min(validation_losses)
        best_epoch = validation_losses.index(best_validation_loss) + 1
        final_epoch = len(epoch_metrics)

        final_loss_gap = validation_losses[-1] - train_losses[-1]
        validation_loss_rise = validation_losses[-1] - best_validation_loss

        # Overfitting: training improves while validation becomes worse.
        if (
            best_epoch < final_epoch
            and validation_loss_rise >= 0.03
            and final_loss_gap >= 0.10
        ):
            severity = "high" if final_loss_gap >= 0.20 else "medium"

            findings.append(
                make_finding(
                    finding_type="overfitting",
                    severity=severity,
                    title="Possible overfitting detected",
                    evidence={
                        "best_validation_epoch": best_epoch,
                        "best_validation_loss": round(best_validation_loss, 4),
                        "final_validation_loss": round(validation_losses[-1], 4),
                        "final_training_loss": round(train_losses[-1], 4),
                        "final_loss_gap": round(final_loss_gap, 4),
                    },
                    recommendations=[
                        f"Use early stopping near epoch {best_epoch}.",
                        "Add dropout or data augmentation.",
                        "Consider reducing model complexity.",
                    ],
                )
            )

        # Underfitting: even training accuracy remains weak.
        if train_accuracies[-1] < 0.65 and final_epoch >= 5:
            findings.append(
                make_finding(
                    finding_type="underfitting",
                    severity="medium",
                    title="Model may be underfitting",
                    evidence={
                        "final_training_accuracy": round(train_accuracies[-1], 4),
                        "final_validation_accuracy": round(
                            validation_accuracies[-1], 4
                        ),
                        "epochs_completed": final_epoch,
                    },
                    recommendations=[
                        "Train for more epochs.",
                        "Increase model capacity with more filters or layers.",
                        "Check whether the learning rate is too small.",
                    ],
                )
            )

        # Plateau: validation accuracy barely changes over recent epochs.
        if len(validation_accuracies) >= 4:
            recent_accuracy_change = (
                max(validation_accuracies[-3:]) - min(validation_accuracies[-3:])
            )

            if recent_accuracy_change <= 0.003:
                findings.append(
                    make_finding(
                        finding_type="plateau",
                        severity="low",
                        title="Validation performance has plateaued",
                        evidence={
                            "recent_validation_accuracy_range": round(
                                recent_accuracy_change, 4
                            ),
                            "final_validation_accuracy": round(
                                validation_accuracies[-1], 4
                            ),
                            "epochs_checked": [final_epoch - 2, final_epoch],
                        },
                        recommendations=[
                            "Consider early stopping.",
                            "Try a learning-rate schedule.",
                            "Test a different model architecture.",
                        ],
                    )
                )

    # Per-class weakness.
    if class_metrics:
        weakest_class = min(class_metrics, key=lambda metric: metric["recall"])

        if weakest_class["recall"] < 0.70:
            severity = "high" if weakest_class["recall"] < 0.50 else "medium"

            findings.append(
                make_finding(
                    finding_type="weak_class_recall",
                    severity=severity,
                    title=f"Weak recall for '{weakest_class['class_name']}'",
                    evidence={
                        "class_name": weakest_class["class_name"],
                        "recall": round(weakest_class["recall"], 4),
                        "precision": round(weakest_class["precision"], 4),
                        "f1_score": round(weakest_class["f1_score"], 4),
                        "support": weakest_class["support"],
                    },
                    recommendations=[
                        "Inspect examples from this class.",
                        "Use augmentation targeted at visually similar classes.",
                        "Consider collecting more examples for this class.",
                    ],
                )
            )

    # Strong class confusion.
    if most_confused_pairs:
        worst_pair = most_confused_pairs[0]

        if worst_pair["count"] >= 50:
            findings.append(
                make_finding(
                    finding_type="class_confusion",
                    severity="medium",
                    title="Frequent class confusion detected",
                    evidence={
                        "actual_class": worst_pair["actual_class"],
                        "predicted_class": worst_pair["predicted_class"],
                        "confusion_count": worst_pair["count"],
                    },
                    recommendations=[
                        "Inspect wrongly classified examples from this pair.",
                        "Use targeted augmentation for these visual categories.",
                        "Try a deeper CNN or transfer-learning model later.",
                    ],
                )
            )

    return findings