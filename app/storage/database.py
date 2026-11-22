import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "experiments.db"


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error_message TEXT,
                dataset_hash TEXT,
                git_commit TEXT,
                tf_version TEXT,
                artifact_directory TEXT
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                validation_loss REAL,
                train_accuracy REAL,
                validation_accuracy REAL,
                learning_rate REAL,
                epoch_time_sec REAL,
                UNIQUE(experiment_id, epoch)
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                experiment_id TEXT PRIMARY KEY,
                test_loss REAL NOT NULL,
                test_accuracy REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS classification_evaluations (
                experiment_id TEXT PRIMARY KEY,
                class_names_json TEXT NOT NULL,
                confusion_matrix_json TEXT NOT NULL,
                class_metrics_json TEXT NOT NULL,
                most_confused_pairs_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                finding_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                rule_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS explanations (
                experiment_id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                explanation_json TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                grounding_passed INTEGER NOT NULL,
                prompt_version TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                latency_ms INTEGER NOT NULL,
                generated_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                validation_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
    """
    CREATE TABLE IF NOT EXISTS tabular_experiments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        target_column TEXT NOT NULL,
        task_type TEXT NOT NULL,
        epochs INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,
        learning_rate REAL NOT NULL,
        seed INTEGER NOT NULL,
        status TEXT NOT NULL,
        error_message TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        artifact_path TEXT
    )
    """
)

connection.execute(
    """
    CREATE TABLE IF NOT EXISTS tabular_epoch_metrics (
        experiment_id TEXT NOT NULL,
        epoch INTEGER NOT NULL,
        train_loss REAL,
        validation_loss REAL,
        train_score REAL,
        validation_score REAL,
        PRIMARY KEY (experiment_id, epoch)
    )
    """
)

connection.execute(
    """
    CREATE TABLE IF NOT EXISTS tabular_evaluations (
        experiment_id TEXT PRIMARY KEY,
        metrics_json TEXT NOT NULL
    )
    """
)

        connection.commit()