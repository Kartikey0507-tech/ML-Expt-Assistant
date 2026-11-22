from datetime import datetime

from pydantic import BaseModel


class DatasetResponse(BaseModel):
    id: str
    name: str
    original_filename: str
    file_type: str
    size_bytes: int
    sha256: str
    status: str
    validation: dict
    created_at: datetime