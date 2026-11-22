import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.datasets.tabular_profile import profile_csv

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.datasets.storage import save_uploaded_file
from app.schemas.dataset import DatasetResponse
from app.storage.database import get_connection



router = APIRouter(prefix="/datasets", tags=["datasets"])

MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def row_to_response(row) -> DatasetResponse:
    return DatasetResponse(
        id=row["id"],
        name=row["name"],
        original_filename=row["original_filename"],
        file_type=row["file_type"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        status=row["status"],
        validation=json.loads(row["validation_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


@router.post(
    "/upload",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    name: str = Form(...),
    file: UploadFile = File(...),
) -> DatasetResponse:
    """Upload and register a CSV or ZIP dataset."""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A dataset file is required.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Dataset exceeds the 200 MB upload limit.",
        )

    dataset_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc)

    try:
        saved = save_uploaded_file(
            dataset_id=dataset_id,
            original_filename=file.filename,
            file_bytes=file_bytes,
        )
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not inspect dataset: {error}",
        )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO datasets (
                id, name, original_filename, file_type, stored_path,
                size_bytes, sha256, status, validation_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                name,
                file.filename,
                saved["file_type"],
                saved["stored_path"],
                saved["size_bytes"],
                saved["sha256"],
                "validated" if saved["validation"]["valid"] else "invalid",
                json.dumps(saved["validation"]),
                created_at.isoformat(),
            ),
        )
        connection.commit()

        row = connection.execute(
            "SELECT * FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()

    return row_to_response(row)


@router.get("/", response_model=list[DatasetResponse])
def list_datasets() -> list[DatasetResponse]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC"
        ).fetchall()

    return [row_to_response(row) for row in rows]

@router.get("/{dataset_id}/profile")
def get_csv_profile(dataset_id: str):
    connection = get_connection()

    try:
        dataset = connection.execute(
            "SELECT * FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()
    finally:
        connection.close()

    if dataset is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    if dataset["file_type"] != "csv":
        raise HTTPException(
            status_code=400,
            detail="CSV profiling is currently available only for CSV datasets.",
        )

    try:
        return profile_csv(dataset["stored_path"])
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error