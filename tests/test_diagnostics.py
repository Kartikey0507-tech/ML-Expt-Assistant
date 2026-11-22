from app.diagnostics.analyzer import analyze_experiment


def make_epoch(
    epoch: int,
    train_loss: float,
    validation_loss: float,
    train_accuracy: float,
    validation_accuracy: float,
) -> dict:
    """Create a small metric record for a synthetic test run."""
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
    }


def finding_types(findings: list[dict]) -> set[str]:
    """Return only diagnostic names, making assertions easy to read."""
    return {finding["finding_type"] for finding in findings}


def test_detects_overfitting():
    """Validation loss worsens after epoch 3 while training improves."""
    metrics = [
        make_epoch(1, 0.90, 0.85, 0.60, 0.63),
        make_epoch(2, 0.70, 0.68, 0.70, 0.71),
        make_epoch(3, 0.50, 0.55, 0.80, 0.78),
        make_epoch(4, 0.30, 0.68, 0.90, 0.76),
        make_epoch(5, 0.15, 0.78, 0.96, 0.73),
    ]

    findings = analyze_experiment(
        epoch_metrics=metrics,
        class_metrics=[],
        most_confused_pairs=[],
    )

    assert "overfitting" in finding_types(findings)

    overfitting = next(
        finding
        for finding in findings
        if finding["finding_type"] == "overfitting"
    )

    assert overfitting["evidence"]["best_validation_epoch"] == 3
    assert overfitting["severity"] == "high"


def test_detects_underfitting():
    """Training accuracy remains below the underfitting threshold."""
    metrics = [
        make_epoch(1, 2.10, 2.05, 0.20, 0.21),
        make_epoch(2, 2.00, 1.98, 0.30, 0.31),
        make_epoch(3, 1.92, 1.90, 0.40, 0.39),
        make_epoch(4, 1.85, 1.84, 0.50, 0.49),
        make_epoch(5, 1.78, 1.77, 0.60, 0.59),
    ]

    findings = analyze_experiment(
        epoch_metrics=metrics,
        class_metrics=[],
        most_confused_pairs=[],
    )

    assert "underfitting" in finding_types(findings)


def test_detects_validation_plateau():
    """Validation accuracy barely changes over the last three epochs."""
    metrics = [
        make_epoch(1, 1.00, 0.95, 0.50, 0.55),
        make_epoch(2, 0.85, 0.82, 0.60, 0.65),
        make_epoch(3, 0.75, 0.74, 0.70, 0.80),
        make_epoch(4, 0.70, 0.70, 0.73, 0.801),
        make_epoch(5, 0.68, 0.69, 0.75, 0.802),
        make_epoch(6, 0.67, 0.68, 0.76, 0.803),
    ]

    findings = analyze_experiment(
        epoch_metrics=metrics,
        class_metrics=[],
        most_confused_pairs=[],
    )

    assert "plateau" in finding_types(findings)


def test_detects_weak_class_recall():
    """A class recall below 0.70 should be reported."""
    class_metrics = [
        {
            "class_name": "T-shirt/top",
            "precision": 0.88,
            "recall": 0.90,
            "f1_score": 0.89,
            "support": 1000,
        },
        {
            "class_name": "Shirt",
            "precision": 0.51,
            "recall": 0.42,
            "f1_score": 0.46,
            "support": 1000,
        },
    ]

    findings = analyze_experiment(
        epoch_metrics=[],
        class_metrics=class_metrics,
        most_confused_pairs=[],
    )

    assert "weak_class_recall" in finding_types(findings)

    weak_class = next(
        finding
        for finding in findings
        if finding["finding_type"] == "weak_class_recall"
    )

    assert weak_class["evidence"]["class_name"] == "Shirt"
    assert weak_class["severity"] == "high"


def test_detects_frequent_class_confusion():
    """A class pair confused 50 or more times should be reported."""
    confused_pairs = [
        {
            "actual_class": "Shirt",
            "predicted_class": "T-shirt/top",
            "count": 84,
        }
    ]

    findings = analyze_experiment(
        epoch_metrics=[],
        class_metrics=[],
        most_confused_pairs=confused_pairs,
    )

    assert "class_confusion" in finding_types(findings)