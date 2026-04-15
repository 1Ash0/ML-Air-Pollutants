from __future__ import annotations

import argparse
import ctypes
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
np.seterr(all="ignore")
# pandas 3.0 introduced StringDtype as the default for object columns.
# Disable this globally so numpy dtype checks (issubdtype, etc.) keep working.
try:
    pd.options.future.infer_string = False  # pandas 3.0+
except AttributeError:
    pass  # pandas < 3.0 doesn't have this option — safe to ignore


VARIABLE_ROW_RE = re.compile(r"(?i)^\s*date\s*&\s*time\s*$")

# Replace if cell contains any of these substrings (case-insensitive).
STATUS_SUBSTRINGS = [
    "maint",
    "calib",
    "power",
    "link",
    "warm",
    "ref",
    "fail",
    "off",
]

# Replace if cell (after strip/lower) equals one of these.
NULL_LITERALS = {"", "nan", "na", "null", "none"}


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def _setup_file_logging(log_file: Path) -> None:
    """Add a file handler so progress is visible even if the terminal buffers output."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.getLogger().level)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(fh)


def _system_memory_percent() -> float:
    """
    Returns system-wide memory usage percent. Uses psutil if available, else
    Windows GlobalMemoryStatusEx via ctypes.
    """
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().percent)
    except Exception:
        pass

    # Windows fallback (this environment is on Windows paths)
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return 0.0
    return float(stat.dwMemoryLoad)


def _stop_if_memory_high(threshold_percent: float) -> None:
    mem = _system_memory_percent()
    if mem >= threshold_percent:
        raise SystemExit(f"Stopping: system memory usage {mem:.1f}% >= {threshold_percent:.1f}%")


def _iter_excel_files(dataset_root: Path) -> Iterable[Path]:
    exts = {".xls", ".xlsx", ".xlsm"}
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() in exts:
            yield path


def _station_from_path(dataset_root: Path, file_path: Path) -> str:
    try:
        rel = file_path.relative_to(dataset_root)
        top = rel.parts[0] if rel.parts else "UNKNOWN"
    except Exception:
        top = file_path.parent.name or "UNKNOWN"

    raw = top.upper()
    # Common folder patterns in this workspace: "Bhatagaon DCR", "DCR AIIMS", "IGKV DCR", "SILTARA DCR"
    for key in ["BHATAGAON", "AIIMS", "IGKV", "SILTARA"]:
        if key in raw:
            return key

    tokens = re.findall(r"[A-Z]+", raw)
    if tokens:
        # Prefer the longest token that isn't "DCR"
        tokens = [t for t in tokens if t != "DCR"]
        if tokens:
            return max(tokens, key=len)
    return raw.strip().replace(" ", "_") or "UNKNOWN"


def _engine_for_excel(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xlsm"}:
        return "openpyxl"
    if ext == ".xls":
        return "xlrd"
    return None


def _find_variable_row(preview: pd.DataFrame) -> int | None:
    if preview.shape[1] == 0:
        return None
    col0 = preview.iloc[:, 0]
    for i, v in enumerate(col0.tolist()):
        if isinstance(v, str) and VARIABLE_ROW_RE.match(v):
            return i
    return None


def _last_nonblank_col_index(preview: pd.DataFrame, up_to_row: int) -> int:
    """
    Returns the last column index (0-based) that has any non-blank value within
    rows [0..up_to_row], inclusive. Falls back to 0.
    """
    sub = preview.iloc[: up_to_row + 1, :]

    # NOTE:
    # Avoid in-place boolean ops on `mask.iloc[:, c]` here. Some pandas builds
    # can throw `AttributeError: 'Series' object has no attribute '_hasna'`
    # during inplace column assignment. This explicit scan is stable.
    last = -1
    for c in range(sub.shape[1]):
        series = sub.iloc[:, c]
        nonblank = series.notna()
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            s = series.astype("string")
            nonblank = nonblank & s.str.strip().ne("").fillna(False)
        if bool(nonblank.any()):
            last = c

    return int(last) if last >= 0 else 0


def _make_unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        key = n
        if key in seen:
            seen[key] += 1
            key = f"{n}__{seen[n]}"
        else:
            seen[key] = 1
        out.append(key)
    return out


def _sanitize_var_name(value: object, fallback: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    s = str(value).strip()
    if s == "":
        return fallback
    # Keep "exact string values" but normalize whitespace to avoid accidental duplicates
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_timestamps(series: pd.Series) -> pd.Series:
    # If already datetime-like, keep.
    # Use pd.api.types instead of np.issubdtype — pandas 3.0+ uses extension
    # dtypes (StringDtype, ArrowDtype) that np.issubdtype cannot handle.
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    s = series.copy()

    # Handle Excel serial dates (common for datetime cells exported as numbers)
    numeric = pd.to_numeric(s, errors="coerce")
    excel_like = numeric.notna() & (numeric > 20000) & (numeric < 70000)
    
    parsed_excel_valid = pd.to_datetime(numeric[excel_like], unit="D", origin="1899-12-30", errors="coerce")
    parsed_excel = pd.Series(pd.NaT, index=s.index)
    parsed_excel.update(parsed_excel_valid)

    # Parse string timestamps.
    # We try a small set of explicit formats first (faster + avoids the pandas
    # "Could not infer format" warning), then fall back to general parsing.
    text = s.astype("string").str.strip()
    text = text.str.replace(r"^(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s+\d{1,2}:\d{2})$", r"\1:00", regex=True)

    parsed_text = pd.to_datetime(text, errors="coerce", format="%d-%m-%Y %H:%M:%S")
    parsed_text = parsed_text.fillna(pd.to_datetime(text, errors="coerce", format="%d/%m/%Y %H:%M:%S"))
    parsed_text = parsed_text.fillna(pd.to_datetime(text, errors="coerce", format="%d-%m-%Y %H:%M"))
    parsed_text = parsed_text.fillna(pd.to_datetime(text, errors="coerce", format="%d/%m/%Y %H:%M"))
    parsed_text = parsed_text.fillna(pd.to_datetime(text, errors="coerce", format="%d-%m-%Y"))
    parsed_text = parsed_text.fillna(pd.to_datetime(text, errors="coerce", format="%d/%m/%Y"))
    # ISO-style dates (YYYY-MM-DD ...) — must be tried WITHOUT dayfirst to avoid conflict warning
    parsed_text = parsed_text.fillna(pd.to_datetime(text, errors="coerce", format="ISO8601"))
    # Final catch-all: mixed formats, dayfirst=True for ambiguous DD/MM values
    parsed_text = parsed_text.fillna(pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True))

    # Prefer parsed_text, fall back to excel serial
    out = parsed_text.fillna(parsed_excel)
    return out


def _clean_value_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # NOTE:
    # Avoid DataFrame.replace(regex=...) here: with some pandas builds it can
    # trigger internal `Series._hasna` AttributeErrors. We do explicit, stable,
    # per-column string masking instead.
    # Use a non-capturing group to avoid pandas warnings about match groups.
    status_re = re.compile(r"(?:" + "|".join(map(re.escape, STATUS_SUBSTRINGS)) + r")", flags=re.IGNORECASE)

    out = df.copy()
    for c in out.columns:
        s = out[c]
        # Convert to standard Python str dtype. In pandas 3.0+ default dtype
        # is StringDtype which returns BooleanDtype masks — cast to plain bool
        # numpy arrays to avoid |  operator type errors.
        s_str = s.astype("object").astype(str)
        s_norm = s_str.str.strip().str.lower()

        # Exact null literals (e.g., "na", "null", "", "none") → NaN
        null_mask = s_norm.isin(NULL_LITERALS).to_numpy(dtype=bool)

        # Partial status tokens (e.g., "Maint.", "Power Off", "Link Fail") → NaN
        status_mask = s_str.str.contains(status_re, na=False).to_numpy(dtype=bool)

        combined_mask = null_mask | status_mask
        cleaned = s.copy()
        cleaned[combined_mask] = np.nan
        out[c] = pd.to_numeric(cleaned, errors="coerce")

    return out


@dataclass
class SheetIngestResult:
    rows: int
    cols: int
    station: str
    file: str
    sheet: str


def ingest_one_sheet(
    excel_path: Path,
    sheet_name: str,
    station: str,
    preview_rows: int,
    memory_threshold_percent: float,
) -> tuple[pd.DataFrame | None, SheetIngestResult | None]:
    engine = _engine_for_excel(excel_path)
    if engine is None:
        logging.warning("Skipping unsupported file: %s", excel_path)
        return None, None

    try:
        preview = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            header=None,
            nrows=preview_rows,
            engine=engine,
        )
    except Exception as e:
        logging.warning("Preview read failed: file=%s sheet=%s err=%s", excel_path, sheet_name, e)
        return None, None

    var_row = _find_variable_row(preview)
    if var_row is None:
        logging.info("No variable-name row found; skipping: file=%s sheet=%s", excel_path, sheet_name)
        return None, None

    last_col = _last_nonblank_col_index(preview, up_to_row=var_row)
    ncols = last_col + 1
    if ncols < 2:
        logging.info("Too few columns after detection; skipping: file=%s sheet=%s", excel_path, sheet_name)
        return None, None

    # Extract variable names from the variable row (columns 1..ncols-1)
    raw_names = preview.iloc[var_row, 1:ncols].tolist()
    value_names = [_sanitize_var_name(v, fallback=f"COL_{i+2}") for i, v in enumerate(raw_names)]
    value_names = _make_unique_names(value_names)

    col_names = ["timestamp"] + value_names

    try:
        # Skip units row (var_row + 1) by starting at var_row + 2
        data = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            header=None,
            skiprows=var_row + 2,
            usecols=list(range(ncols)),
            names=col_names,
            engine=engine,
        )
    except Exception as e:
        logging.warning("Data read failed: file=%s sheet=%s err=%s", excel_path, sheet_name, e)
        return None, None

    if data.empty:
        return None, None

    # Parse timestamps
    data["timestamp"] = _parse_timestamps(data["timestamp"])
    data = data.dropna(subset=["timestamp"])

    # Clean values
    value_cols = [c for c in data.columns if c != "timestamp"]
    data[value_cols] = _clean_value_frame(data[value_cols])

    # Add metadata
    data["station"] = station
    data["source_file"] = str(excel_path)
    data["source_sheet"] = str(sheet_name)

    # Final ordering
    data = data.sort_values(["station", "timestamp"]).reset_index(drop=True)

    _stop_if_memory_high(memory_threshold_percent)

    result = SheetIngestResult(
        rows=int(data.shape[0]),
        cols=int(data.shape[1]),
        station=station,
        file=str(excel_path),
        sheet=str(sheet_name),
    )
    return data, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest messy air-quality Excel files into a tidy 15-min Parquet table.")
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--out-parquet", type=str, default=str(Path("artifacts") / "data_15min.parquet"))
    parser.add_argument(
        "--chunk-dir",
        type=str,
        default=str(Path("artifacts") / "_ingest_chunks"),
        help="Directory for streaming parquet chunks (safer for large datasets).",
    )
    parser.add_argument(
        "--stream-chunks",
        action="store_true",
        help="Write each ingested sheet to chunk parquet files and merge at the end (recommended).",
    )
    parser.add_argument(
        "--no-stream-chunks",
        action="store_true",
        help="Disable streaming chunk writes and keep all sheets in memory (not recommended for large datasets).",
    )
    parser.add_argument("--preview-rows", type=int, default=50)
    parser.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    parser.add_argument("--max-sheets-per-file", type=int, default=0, help="0 = no limit")
    parser.add_argument(
        "--sheet-name-regex",
        type=str,
        default=r"^\d{1,2}$",
        help="Only process sheets whose names match this regex. Use '.*' to process all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing chunk directory instead of deleting it (streaming mode only).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(Path("artifacts") / "ingest_run.log"),
        help="Write progress logs to this file.",
    )
    parser.add_argument("--memory-threshold", type=float, default=80.0, help="Stop if system memory percent >= this value.")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    _setup_file_logging(Path(args.log_file))

    dataset_root = Path(args.dataset_root).resolve()
    out_path = Path(args.out_parquet).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = Path(args.chunk_dir).resolve()
    stream_chunks = bool(args.stream_chunks) and not bool(args.no_stream_chunks)
    if not args.stream_chunks and not args.no_stream_chunks:
        # Default to streaming for safety on larger datasets.
        stream_chunks = True

    excel_files = sorted(_iter_excel_files(dataset_root), key=lambda p: p.as_posix())
    if args.max_files and args.max_files > 0:
        excel_files = excel_files[: args.max_files]

    if not excel_files:
        logging.error("No Excel files found under: %s", dataset_root)
        return 2

    logging.info("Found %d Excel files under %s", len(excel_files), dataset_root)

    frames: list[pd.DataFrame] = []
    ingested = 0
    skipped_sheets = 0

    if stream_chunks:
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401
        except Exception as e:
            logging.error(
                "Streaming chunk mode requires `pyarrow`. Install it first or pass --no-stream-chunks. err=%s",
                e,
            )
            return 5

        if chunk_dir.exists() and not args.resume:
            shutil.rmtree(chunk_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        # Resume chunk index if requested.
        existing = sorted(chunk_dir.glob("part_*.parquet"))
        chunk_idx = int(len(existing)) if args.resume else 0
        if args.resume and existing:
            logging.info("Resuming: found %d existing chunk files in %s", len(existing), chunk_dir)

    sheet_name_re = re.compile(args.sheet_name_regex)

    for fidx, excel_path in enumerate(excel_files, start=1):
        file_t0 = time.time()
        station = _station_from_path(dataset_root, excel_path)
        engine = _engine_for_excel(excel_path)
        if engine is None:
            continue

        logging.info("(%d/%d) File: %s | station=%s", fidx, len(excel_files), excel_path, station)

        try:
            xl = pd.ExcelFile(excel_path, engine=engine)
            sheet_names = xl.sheet_names
        except Exception as e:
            logging.warning("Failed to open workbook; skipping file=%s err=%s", excel_path, e)
            continue

        if args.max_sheets_per_file and args.max_sheets_per_file > 0:
            sheet_names = sheet_names[: args.max_sheets_per_file]

        for sidx, sheet in enumerate(sheet_names, start=1):
            # Most workbooks contain extra sheets like "CALIBRATION REPORT", "POWER OFF".
            # Skipping non-matching sheet names prevents pointless preview reads and
            # speeds up ingestion substantially at scale.
            if not sheet_name_re.match(str(sheet)):
                continue
            try:
                df, meta = ingest_one_sheet(
                    excel_path=excel_path,
                    sheet_name=sheet,
                    station=station,
                    preview_rows=args.preview_rows,
                    memory_threshold_percent=float(args.memory_threshold),
                )
            except SystemExit:
                raise
            except Exception as e:
                import traceback
                logging.warning("Sheet ingest crashed; skipping file=%s sheet=%s err=%s\n%s", 
                                excel_path, sheet, e, traceback.format_exc())
                df, meta = None, None

            if df is None or meta is None:
                skipped_sheets += 1
                continue

            if stream_chunks:
                # Write immediately to a chunk file to keep memory bounded.
                chunk_idx += 1
                chunk_path = chunk_dir / f"part_{chunk_idx:06d}.parquet"
                try:
                    df.to_parquet(chunk_path, index=False)
                except Exception as e:
                    logging.error("Failed writing chunk parquet: %s err=%s", chunk_path, e)
                    return 6
            else:
                frames.append(df)
            ingested += 1

            if ingested % 25 == 0:
                mem = _system_memory_percent()
                logging.info("Progress: ingested_sheets=%d skipped_sheets=%d mem=%.1f%%", ingested, skipped_sheets, mem)
                _stop_if_memory_high(float(args.memory_threshold))

        logging.info(
            "Finished file %d/%d in %.1fs | ingested_sheets=%d skipped_sheets=%d",
            fidx,
            len(excel_files),
            time.time() - file_t0,
            ingested,
            skipped_sheets,
        )

    if stream_chunks:
        # Merge chunk files into a single parquet without loading all data at once.
        from pyarrow import Table, concat_tables  # type: ignore
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        chunk_files = sorted(chunk_dir.glob("part_*.parquet"))
        if not chunk_files:
            logging.error("No chunk parquets written. Check logs and variable-row detection.")
            return 3

        logging.info("Merging %d chunk parquets into %s ...", len(chunk_files), out_path)
        _stop_if_memory_high(float(args.memory_threshold))

        # 1) Build a union schema from parquet metadata (no full reads).
        schemas: list[pa.Schema] = []
        for p in chunk_files:
            pf = pq.ParquetFile(p)
            schemas.append(pf.schema_arrow)

        union_fields: dict[str, pa.Field] = {}
        for sch in schemas:
            for field in sch:
                # First occurrence wins; later fields with same name are assumed compatible.
                union_fields.setdefault(field.name, field)

        # Prefer stable ordering: station, timestamp first, then others alphabetically.
        preferred = ["station", "timestamp", "source_file", "source_sheet"]
        rest = sorted([n for n in union_fields.keys() if n not in preferred])
        ordered_names = [n for n in preferred if n in union_fields] + rest
        union_schema = pa.schema([union_fields[n] for n in ordered_names])

        # 2) Stream tables into a ParquetWriter, casting/adding missing columns per chunk.
        if out_path.exists():
            out_path.unlink()

        writer: pq.ParquetWriter | None = None
        total_rows = 0
        for p in chunk_files:
            table = pq.read_table(p)
            # Add any missing fields as nulls.
            missing = [name for name in union_schema.names if name not in table.schema.names]
            if missing:
                for name in missing:
                    table = table.append_column(name, pa.nulls(table.num_rows, type=union_schema.field(name).type))
            # Reorder/cast columns to union schema
            table = table.select(union_schema.names)
            try:
                table = table.cast(union_schema, safe=False)
            except Exception:
                # As a fallback, keep as-is (writer will error if incompatible). This is rare.
                pass

            if writer is None:
                writer = pq.ParquetWriter(out_path, union_schema)
            writer.write_table(table)
            total_rows += int(table.num_rows)
            _stop_if_memory_high(float(args.memory_threshold))

        if writer is not None:
            writer.close()

        logging.info("Wrote merged parquet: %s (rows=%d)", out_path, total_rows)
    else:
        if not frames:
            logging.error("No sheets ingested successfully. Check logs and the variable-row regex.")
            return 3

        logging.info("Concatenating %d sheet tables...", len(frames))
        _stop_if_memory_high(float(args.memory_threshold))

        full = pd.concat(frames, ignore_index=True, sort=True)
        full = full.sort_values(["station", "timestamp"]).reset_index(drop=True)

        # Save
        logging.info("Writing parquet: %s (rows=%d, cols=%d)", out_path, full.shape[0], full.shape[1])
        try:
            full.to_parquet(out_path, index=False)
        except Exception as e:
            logging.error(
                "Failed to write parquet. Ensure `pyarrow` is installed in the active env. err=%s",
                e,
            )
            return 4

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
