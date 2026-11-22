import csv
import hashlib
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "uploaded_datasets"


def inspect_csv(file_path: Path) -> dict:
    """Read only a small CSV preview; do not load the full dataset yet."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        return {
            "valid": False,
            "message": "The CSV file is empty.",
        }

    headers = rows[0]

    return {
        "valid": len(headers) >= 2,
        "kind": "csv",
        "column_count": len(headers),
        "columns": headers,
        "preview_rows": rows[1:6],
        "message": (
            "CSV uploaded. Later you will choose whether it is "
            "tabular or text data and select its target column."
        ),
    }


def inspect_zip(file_path: Path) -> dict:
    """Inspect ZIP contents without extracting potentially unsafe files."""
    with zipfile.ZipFile(file_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if not name.endswith("/")
        ]

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_names = [
        name
        for name in names
        if Path(name).suffix.lower() in image_extensions
    ]

    class_names = sorted(
        {
            Path(name).parts[-2]
            for name in image_names
            if len(Path(name).parts) >= 2
        }
    )

    return {
        "valid": len(image_names) >= 2 and len(class_names) >= 2,
        "kind": "zip",
        "file_count": len(names),
        "image_count": len(image_names),
        "detected_class_folders": class_names,
        "message": (
            "ZIP uploaded. Image-folder validation will be completed "
            "when the custom image trainer is added."
        ),
    }


def inspect_dataset(file_path: Path, file_type: str) -> dict:
    if file_type == "csv":
        return inspect_csv(file_path)

    if file_type == "zip":
        return inspect_zip(file_path)

    return {
        "valid": False,
        "message": "Unsupported dataset type.",
    }


def save_uploaded_file(
    dataset_id: str,
    original_filename: str,
    file_bytes: bytes,
) -> dict:
    """Save one uploaded file under a generated, safe directory."""
    extension = Path(original_filename).suffix.lower()

    if extension not in {".csv", ".zip"}:
        raise ValueError("Only CSV and ZIP dataset files are supported.")

    file_type = "csv" if extension == ".csv" else "zip"

    dataset_directory = DATASET_ROOT / dataset_id
    dataset_directory.mkdir(parents=True, exist_ok=True)

    stored_path = dataset_directory / f"source{extension}"
    stored_path.write_bytes(file_bytes)

    validation = inspect_dataset(stored_path, file_type)

    return {
        "file_type": file_type,
        "stored_path": str(stored_path),
        "size_bytes": len(file_bytes),
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
        "validation": validation,
    }