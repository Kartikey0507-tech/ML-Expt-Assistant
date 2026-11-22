from pathlib import Path

import pandas as pd

PREVIEW_ROWS = 5000
DISPLAY_ROWS = 10
EXAMPLE_VALUES = 5


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    last_error = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, nrows=PREVIEW_ROWS), encoding
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(
        "The CSV could not be read using UTF-8 or Latin-1 encoding."
    ) from last_error


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    return "text_or_category"


def profile_csv(path: str) -> dict:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError("The uploaded CSV file no longer exists.")

    dataframe, encoding = _read_csv(csv_path)

    if dataframe.empty and len(dataframe.columns) == 0:
        raise ValueError("This CSV is empty or has no readable columns.")

    columns = []

    for column_name in dataframe.columns:
        series = dataframe[column_name]
        non_null_values = series.dropna()

        examples = [
            str(value)
            for value in non_null_values.astype(str).head(EXAMPLE_VALUES).tolist()
        ]

        columns.append(
            {
                "name": str(column_name),
                "inferred_type": _infer_column_type(series),
                "pandas_dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(
                    float(series.isna().mean() * 100), 2
                ) if len(series) else 0.0,
                "unique_count": int(series.nunique(dropna=True)),
                "example_values": examples,
            }
        )

    preview = dataframe.head(DISPLAY_ROWS).where(
        pd.notna(dataframe.head(DISPLAY_ROWS)), None
    )

    return {
        "encoding": encoding,
        "sampled_rows": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": columns,
        "preview": preview.to_dict(orient="records"),
        "note": (
            f"Showing the first {len(dataframe)} rows for inspection. "
            "Training will later read the complete CSV."
        ),
    }