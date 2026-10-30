"""
Structured data extractor — CSV, XLSX, JSON, Parquet.

Returns a lightweight metadata summary instead of extracting all content.
The actual data analysis happens via Code Interpreter at query time.
Permanent ingestion to PostgreSQL happens async via /ingest endpoint.
"""

import io
import logging
from typing import Optional

logger = logging.getLogger("haystack-api.extract.structured")


async def extract_structured(filename: str, ext: str, file_bytes: bytes) -> str:
    """
    Generate a fast metadata summary of a structured data file.
    Does NOT parse every row — just enough to describe the file.
    """
    try:
        if ext in ("csv", "tsv"):
            return _summarize_csv(filename, file_bytes, delimiter="\t" if ext == "tsv" else ",")
        elif ext in ("xlsx", "xls"):
            return _summarize_excel(filename, file_bytes)
        elif ext in ("json", "jsonl"):
            return _summarize_json(filename, file_bytes)
        elif ext == "parquet":
            return _summarize_parquet(filename, file_bytes)
        else:
            return f"Structured data file: {filename} ({len(file_bytes)} bytes)"
    except Exception as e:
        logger.warning(f"Structured extraction fallback for {filename}: {e}")
        return f"Data file: {filename} ({len(file_bytes):,} bytes). Use code interpreter to analyze."


def _summarize_csv(filename: str, file_bytes: bytes, delimiter: str = ",") -> str:
    """Summarize CSV/TSV by reading header + sample rows + row count."""
    import pandas as pd

    try:
        # Read only first 10 rows for structure
        df_sample = pd.read_csv(io.BytesIO(file_bytes), delimiter=delimiter, nrows=10)

        # Count total rows efficiently
        total_rows = file_bytes.count(b"\n")

        columns = list(df_sample.columns)
        dtypes = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
        sep_name = "tab" if delimiter == "\t" else "comma"

        # Build summary
        lines = [
            f"File: {filename}",
            f"Type: CSV ({sep_name}-separated)",
            f"Rows: ~{total_rows:,}",
            f"Columns ({len(columns)}): {', '.join(columns)}",
            f"Column types: {dtypes}",
            "",
            "Sample data (first 5 rows):",
            df_sample.head().to_string(index=False),
        ]

        # Add basic stats for numeric columns
        numeric_cols = df_sample.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            lines.append("")
            lines.append(f"Numeric columns: {', '.join(numeric_cols)}")

            # Date range. The first row is already parsed in df_sample, and the last
            # comes from a tail byte-slice — re-reading the file would be a second
            # full pass over every row just to reach the end.
            date_cols = [c for c in columns if any(kw in c.lower() for kw in ("date", "time", "timestamp"))]
            if date_cols and len(df_sample) > 0:
                col_idx = columns.index(date_cols[0])
                first_date = df_sample.iloc[0, col_idx]
                last_date = _last_row_field(file_bytes, delimiter, col_idx)
                if last_date:
                    lines.append(f"Date range: {first_date} to {last_date}")

        # Add symbol/instrument if present
        symbol_cols = [c for c in columns if any(kw in c.lower() for kw in ("symbol", "ticker", "instrument"))]
        if symbol_cols:
            symbols = df_sample[symbol_cols[0]].unique().tolist()
            lines.append(f"Symbols: {', '.join(str(s) for s in symbols[:10])}")

        lines.append("")
        lines.append("Use code interpreter with pandas for detailed analysis of this data file.")

        return "\n".join(lines)

    except Exception as e:
        return f"CSV file: {filename} (~{len(file_bytes):,} bytes). Parse error: {e}. Use code interpreter."


def _summarize_excel(filename: str, file_bytes: bytes) -> str:
    """Summarize Excel file — sheet names + sample from first sheet."""
    try:
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
        sheet_names = wb.sheetnames

        lines = [
            f"File: {filename}",
            f"Type: Excel workbook",
            f"Sheets ({len(sheet_names)}): {', '.join(sheet_names)}",
        ]

        # Sample first sheet
        ws = wb[sheet_names[0]]
        rows = list(ws.iter_rows(max_row=6, values_only=True))
        if rows:
            header = [str(c) if c is not None else "" for c in rows[0]]
            lines.append(f"First sheet columns: {', '.join(header)}")
            lines.append(f"First sheet rows: ~{ws.max_row:,}")

        wb.close()
        lines.append("")
        lines.append("Use code interpreter with pandas for detailed analysis.")
        return "\n".join(lines)

    except Exception as e:
        return f"Excel file: {filename} ({len(file_bytes):,} bytes). Use code interpreter."


def _summarize_json(filename: str, file_bytes: bytes) -> str:
    """Summarize JSON — structure, keys, array length."""
    import json

    try:
        text = file_bytes.decode("utf-8")

        # JSONL: count lines, show first entry
        if filename.endswith(".jsonl"):
            lines_list = text.strip().split("\n")
            first = json.loads(lines_list[0])
            return (
                f"File: {filename}\n"
                f"Type: JSON Lines\n"
                f"Entries: {len(lines_list):,}\n"
                f"Keys: {list(first.keys()) if isinstance(first, dict) else 'array'}\n"
                f"Sample: {json.dumps(first, indent=2)[:500]}"
            )

        data = json.loads(text)

        if isinstance(data, list):
            sample = json.dumps(data[0], indent=2)[:500] if data else "[]"
            return (
                f"File: {filename}\n"
                f"Type: JSON array\n"
                f"Items: {len(data):,}\n"
                f"Sample (first item): {sample}"
            )
        elif isinstance(data, dict):
            return (
                f"File: {filename}\n"
                f"Type: JSON object\n"
                f"Top-level keys: {list(data.keys())[:20]}\n"
                f"Preview: {json.dumps(data, indent=2)[:500]}"
            )
        else:
            return f"JSON file: {filename}. Content type: {type(data).__name__}"

    except Exception as e:
        return f"JSON file: {filename} ({len(file_bytes):,} bytes). Parse error: {e}"


def _summarize_parquet(filename: str, file_bytes: bytes) -> str:
    """Summarize Parquet file."""
    try:
        import pandas as pd
        df = pd.read_parquet(io.BytesIO(file_bytes))
        return (
            f"File: {filename}\n"
            f"Type: Parquet\n"
            f"Rows: {len(df):,}\n"
            f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist())}\n"
            f"Sample:\n{df.head().to_string(index=False)}"
        )
    except Exception as e:
        return f"Parquet file: {filename} ({len(file_bytes):,} bytes). Use code interpreter."


def _last_row_field(file_bytes: bytes, delimiter: str, col_idx: int) -> str:
    """
    Read one field from the final data row without re-parsing the file.

    Only the tail of the buffer is touched, so cost is independent of row count.
    csv.reader (not str.split) so quoted fields containing the delimiter survive.
    """
    import csv

    tail = file_bytes[-65536:].rstrip(b"\r\n")
    if b"\n" not in tail:
        return ""  # header-only file, or one row we already have from df_sample
    last_line = tail.rsplit(b"\n", 1)[-1].decode("utf-8", "replace")

    try:
        fields = next(csv.reader([last_line], delimiter=delimiter))
        return fields[col_idx].strip()
    except (StopIteration, IndexError):
        return ""
