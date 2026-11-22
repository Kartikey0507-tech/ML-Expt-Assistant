import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from app.storage.database import get_connection


DATASET_CLASS_NAMES = {
    "fashion_mnist": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "cifar10": [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ],
}


class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, experiment_id: str):
        super().__init__()
        self.experiment_id = experiment_id
        self.epoch_started_at = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started_at = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        learning_rate = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )
        epoch_time_sec = time.perf_counter() - self.epoch_started_at

        with get_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO epoch_metrics (
                    experiment_id, epoch, train_loss, validation_loss,
                    train_accuracy, validation_accuracy,
                    learning_rate, epoch_time_sec
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.experiment_id,
                    epoch + 1,
                    logs.get("loss"),
                    logs.get("val_loss"),
                    logs.get("accuracy"),
                    logs.get("val_accuracy"),
                    learning_rate,
                    epoch_time_sec,
                ),
            )
            connection.commit()


def calculate_dataset_hash(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> str:
    digest = hashlib.sha256()

    for array in (x_train, y_train, x_test, y_test):
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes())

    return digest.hexdigest()


def load_dataset(dataset_name: str):
    """Load and prepare a supported image-classification dataset."""
    if dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )

    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.cifar10.load_data()
        )

        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_hash = calculate_dataset_hash(x_train, y_train, x_test, y_test)

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Fashion-MNIST has no explicit channel dimension.
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        dataset_hash,
        DATASET_CLASS_NAMES[dataset_name],
    )


def build_simple_cnn(
    input_shape: tuple,
    learning_rate: float,
    number_of_classes: int,
) -> tf.keras.Model:
    """Create a CNN that works with both grayscale and RGB images."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(number_of_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(experiment_id: str, config: dict, artifact_directory: Path):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])

    (
        x_train,
        y_train,
        x_test,
        y_test,
        dataset_hash,
        class_names,
    ) = load_dataset(config["dataset_name"])

    x_validation = x_train[-5000:]
    y_validation = y_train[-5000:]
    x_train = x_train[:-5000]
    y_train = y_train[:-5000]

    model = build_simple_cnn(
        input_shape=x_train.shape[1:],
        learning_rate=config["learning_rate"],
        number_of_classes=len(class_names),
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.000001,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        MetricsLogger(experiment_id),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=2,
        callbacks=callbacks,
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    model.save(artifact_directory / "model.keras")

    return model, x_test, y_test, dataset_hash, class_names


def evaluate_and_save(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    artifact_directory: Path,
) -> dict:
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    predicted_probabilities = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predicted_probabilities, axis=1)

    matrix = confusion_matrix(
        y_test,
        predicted_labels,
        labels=range(len(class_names)),
    )

    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_test,
        predicted_labels,
        labels=range(len(class_names)),
        zero_division=0,
    )

    class_metrics = [
        {
            "class_name": class_names[index],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1_score[index]),
            "support": int(support[index]),
        }
        for index in range(len(class_names))
    ]

    pairs = []

    for actual_index in range(len(class_names)):
        for predicted_index in range(len(class_names)):
            count = int(matrix[actual_index][predicted_index])

            if actual_index != predicted_index and count > 0:
                pairs.append(
                    {
                        "actual_class": class_names[actual_index],
                        "predicted_class": class_names[predicted_index],
                        "count": count,
                    }
                )

    summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "tensorflow_version": tf.__version__,
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "class_metrics": class_metrics,
        "most_confused_pairs": sorted(
            pairs,
            key=lambda pair: pair["count"],
            reverse=True,
        )[:5],
    }

    with (artifact_directory / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    return summary