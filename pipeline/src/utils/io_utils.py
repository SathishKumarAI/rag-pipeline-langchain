"""
File I/O utilities for the RAG pipeline.

WHAT: Safe file read/write operations with support for text, JSON, CSV, and Parquet.
WHY: Provides consistent, safe file operations with error handling and atomic writes
     (tmp + rename pattern to prevent corruption).
HOW: Uses pathlib for path handling, implements atomic writes using temporary files,
     and provides helpful path manipulation utilities.

Usage:
    from utils.io_utils import read_text, write_text, read_json, write_json, safe_write
    
    content = read_text("data/raw/document.txt")
    write_text("data/processed/output.txt", content)
    data = read_json("config.json")
    safe_write("output.json", lambda f: json.dump(data, f))
"""

import json
import csv
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd


# def read_text(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
#     """
#     Read text from a file.
    
#     Args:
#         file_path: Path to the file
#         encoding: File encoding (default: utf-8)
        
#     Returns:
#         File contents as string
        
#     Raises:
#         FileNotFoundError: If file doesn't exist
#         IOError: If file cannot be read
#     """
#     file_path = Path(file_path)
#     if not file_path.exists():
#         raise FileNotFoundError(f"File not found: {file_path}")
    
#     with open(file_path, "r", encoding=encoding) as f:
#         return f.read()

from pathlib import Path
from pypdf import PdfReader  # pip install pypdf

def read_text(file_path: Path):
    """
    Safe universal reader:
    - PDFs via pypdf
    - Other text files via multi-encoding fallback
    """
    suffix = file_path.suffix.lower()

    # --- PDF handling ---
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            parts.append(text)
        return "\n".join(parts)

    # --- Generic text handling ---
    encodings = ["utf-8", "latin-1", "utf-16", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    # Final fallback: binary read + ignore bad bytes
    with open(file_path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")


def write_text(
    file_path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    atomic: bool = True,
) -> None:
    """
    Write text to a file (atomically by default).
    
    Args:
        file_path: Path to the file
        content: Text content to write
        encoding: File encoding (default: utf-8)
        atomic: If True, use atomic write (tmp + rename)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if atomic:
        safe_write(file_path, lambda f: f.write(content), encoding=encoding)
    else:
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)


def read_json(file_path: Union[str, Path]) -> Any:
    """
    Read JSON from a file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    file_path: Union[str, Path],
    data: Any,
    indent: int = 2,
    atomic: bool = True,
) -> None:
    """
    Write data as JSON to a file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to serialize as JSON
        indent: JSON indentation (default: 2)
        atomic: If True, use atomic write
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if atomic:
        safe_write(
            file_path,
            lambda f: json.dump(data, f, indent=indent, ensure_ascii=False),
        )
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)


def read_csv(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read CSV file as list of dictionaries.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries (one per row)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(
    file_path: Union[str, Path],
    data: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
    atomic: bool = True,
) -> None:
    """
    Write list of dictionaries to CSV file.
    
    Args:
        file_path: Path to the CSV file
        data: List of dictionaries to write
        fieldnames: CSV column names (if None, inferred from first row)
        atomic: If True, use atomic write
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data:
        return
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    def write_func(f):
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    if atomic:
        safe_write(file_path, write_func)
    else:
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            write_func(f)


def read_parquet(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read Parquet file as pandas DataFrame.
    
    Args:
        file_path: Path to the Parquet file
        
    Returns:
        DataFrame
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_parquet(file_path)


def write_parquet(
    file_path: Union[str, Path],
    df: pd.DataFrame,
    atomic: bool = True,
) -> None:
    """
    Write pandas DataFrame to Parquet file.
    
    Args:
        file_path: Path to the Parquet file
        df: DataFrame to write
        atomic: If True, use atomic write
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if atomic:
        # Parquet doesn't support direct atomic write, so we write to temp then move
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        df.to_parquet(temp_path, index=False)
        temp_path.replace(file_path)
    else:
        df.to_parquet(file_path, index=False)


def safe_write(
    file_path: Union[str, Path],
    write_func: Callable,
    encoding: str = "utf-8",
) -> None:
    """
    Atomically write to a file using temporary file + rename pattern.
    
    This prevents file corruption if the process is interrupted during write.
    
    Args:
        file_path: Path to the final file
        write_func: Function that takes a file handle and writes data
        encoding: File encoding (default: utf-8)
        
    Example:
        safe_write("output.json", lambda f: json.dump(data, f))
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in same directory
    temp_fd, temp_path = tempfile.mkstemp(
        dir=file_path.parent,
        prefix=file_path.name + ".",
        suffix=".tmp",
    )
    
    try:
        with open(temp_fd, "w", encoding=encoding) as f:
            write_func(f)
        
        # Atomic rename
        Path(temp_path).replace(file_path)
    except Exception:
        # Clean up temp file on error
        Path(temp_path).unlink(missing_ok=True)
        raise


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        Path object
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
    """
    return Path(file_path).stat().st_size


def list_files(
    dir_path: Union[str, Path],
    pattern: Optional[str] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    List files in a directory.
    
    Args:
        dir_path: Directory path
        pattern: Glob pattern (e.g., "*.txt")
        recursive: If True, search recursively
        
    Returns:
        List of file paths
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    
    if pattern:
        if recursive:
            return list(dir_path.rglob(pattern))
        else:
            return list(dir_path.glob(pattern))
    else:
        if recursive:
            return [f for f in dir_path.rglob("*") if f.is_file()]
        else:
            return [f for f in dir_path.iterdir() if f.is_file()]

