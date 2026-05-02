"""Structured adapters around the legacy Fumeus file-oriented functions."""

from __future__ import annotations

import ast
import base64
import binascii
import csv
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

ROOT_DIR = Path(__file__).resolve().parents[2]
FUMEUS_MAIN_DIR = ROOT_DIR
VALID_MODES = {"CC", "DRC", "RSV", "RCV"}
VALID_FORMATS = {"csv", "json"}
VALID_UPLOAD_ENCODINGS = {"text", "base64"}


def _load_legacy_module(module_name: str, file_name: str) -> ModuleType:
    if not FUMEUS_MAIN_DIR.exists():
        raise RuntimeError(f"Fumeus source directory not found: {FUMEUS_MAIN_DIR}")

    if str(FUMEUS_MAIN_DIR) not in sys.path:
        sys.path.insert(0, str(FUMEUS_MAIN_DIR))

    module_path = FUMEUS_MAIN_DIR / file_name
    spec = importlib.util.spec_from_file_location(f"_fumeus_legacy_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy Fumeus module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy_generate = _load_legacy_module("generate", "generate.py")
legacy_score = _load_legacy_module("score", "score.py")
legacy_utils = _load_legacy_module("fum_utils", "fum_utils.py")


def _input_file(path: str | Path, label: str) -> str:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return str(resolved)


def _output_path(
    path: str | Path | None,
    output_format: str,
    default_name: str,
) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    suffix = ".json" if output_format == "json" else ".csv"

    if path is not None:
        resolved = Path(path).expanduser()
        if resolved.suffix == "":
            resolved = resolved.with_suffix(suffix)
        return legacy_utils.prepare_output_path(str(resolved), suffix), None

    temp_dir = tempfile.TemporaryDirectory()
    return str(Path(temp_dir.name) / f"{default_name}{suffix}"), temp_dir


def _upload_root(upload_dir: str | Path | None = None) -> Path:
    root = Path(upload_dir or os.getenv("FUMEUS_MCP_UPLOAD_DIR") or ROOT_DIR / "uploads")
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _safe_upload_path(file_name: str, upload_dir: str | Path | None = None) -> Path:
    if not file_name or file_name.strip() == "":
        raise ValueError("file_name must not be empty")

    candidate = Path(file_name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("file_name must be a safe relative path inside the upload directory")

    root = _upload_root(upload_dir)
    resolved = (root / candidate).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("file_name must stay inside the upload directory")
    return resolved


def _normalize_format(output_format: str) -> Literal["csv", "json"]:
    normalized = output_format.lower()
    if normalized not in VALID_FORMATS:
        raise ValueError("output_format must be either 'csv' or 'json'")
    return normalized  # type: ignore[return-value]


def _normalize_mode(mode: str) -> str:
    normalized = mode.upper()
    if normalized not in VALID_MODES:
        raise ValueError("mode must be one of CC, DRC, RSV, or RCV")
    return normalized


def _decode_upload_content(content: str, encoding: str) -> bytes:
    normalized = encoding.lower()
    if normalized not in VALID_UPLOAD_ENCODINGS:
        raise ValueError("encoding must be either 'text' or 'base64'")

    if normalized == "base64":
        try:
            return base64.b64decode(content, validate=True)
        except binascii.Error as exc:
            raise ValueError("content is not valid base64") from exc

    return content.encode("utf-8")


def _read_terms_output(path: str, output_format: str) -> list[dict[str, float | str]]:
    if output_format == "json":
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    terms: list[dict[str, float | str]] = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2:
                continue
            terms.append({"term": row[0], "score": float(row[1])})
    return terms


def _read_scores_output(path: str, output_format: str) -> list[dict[str, Any]]:
    if output_format == "json":
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                terms_found = ast.literal_eval(row[2])
            except (SyntaxError, ValueError):
                terms_found = row[2]
            records.append({"text": row[0], "score": float(row[1]), "terms_found": terms_found})
    return records


def upload_file(
    file_name: str,
    content: str,
    encoding: str = "text",
    overwrite: bool = False,
    upload_dir: str | None = None,
) -> dict[str, Any]:
    """Save user-provided file content into the MCP upload directory."""

    target = _safe_upload_path(file_name, upload_dir)
    existed = target.exists()
    if existed and not overwrite:
        raise FileExistsError(f"Upload already exists: {target}")

    data = _decode_upload_content(content, encoding)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)

    return {
        "path": str(target),
        "file_name": file_name,
        "size_bytes": len(data),
        "encoding": encoding.lower(),
        "overwritten": existed,
    }


def generate_smoke_terms(
    dataset_path: str,
    header: bool = False,
    text_column: int = 0,
    label_column: int = -1,
    ngram_length: int = 1,
    mode: str = "CC",
    top_n: int = 200,
    output_format: str = "json",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Generate ranked smoke terms from a labeled CSV dataset."""

    if ngram_length < 1:
        raise ValueError("ngram_length must be at least 1")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    normalized_format = _normalize_format(output_format)
    normalized_mode = _normalize_mode(mode)
    input_path = _input_file(dataset_path, "dataset_path")
    resolved_output_path, temp_dir = _output_path(output_path, normalized_format, "smoke_terms")

    try:
        legacy_generate.main(
            input_path,
            header=header,
            x_column=text_column,
            y_column=label_column,
            ngram_length=ngram_length,
            mode=normalized_mode,
            output_fname=resolved_output_path,
            top_n=top_n,
            format=normalized_format,
        )
        terms = _read_terms_output(resolved_output_path, normalized_format)
        result: dict[str, Any] = {
            "terms": terms,
            "count": len(terms),
            "mode": normalized_mode,
            "ngram_length": ngram_length,
        }
        if output_path is not None:
            result["output_path"] = resolved_output_path
        return result
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def score_records(
    dataset_path: str,
    terms_path: str,
    header: bool = False,
    terms_header: bool = False,
    text_column: int = 0,
    output_format: str = "json",
    output_path: str | None = None,
) -> dict[str, Any]:
    """Score CSV records with a weighted smoke-term dictionary."""

    normalized_format = _normalize_format(output_format)
    input_path = _input_file(dataset_path, "dataset_path")
    input_terms_path = _input_file(terms_path, "terms_path")
    resolved_output_path, temp_dir = _output_path(output_path, normalized_format, "scored_records")

    try:
        legacy_score.main(
            input_path,
            input_terms_path,
            resolved_output_path,
            header=header,
            header2=terms_header,
            x_column=text_column,
            format=normalized_format,
        )
        records = _read_scores_output(resolved_output_path, normalized_format)
        result: dict[str, Any] = {
            "records": records,
            "count": len(records),
        }
        if output_path is not None:
            result["output_path"] = resolved_output_path
        return result
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def clean_text(texts: list[str], stop_words: list[str] | None = None) -> dict[str, Any]:
    """Return Fumeus-cleaned token lists for the supplied texts."""

    cleaned = legacy_utils.clean_documents(texts, stop_words or [])
    documents = [list(document) for document in cleaned]
    return {"documents": documents, "count": len(documents)}


def score_text(text: str, terms: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one text string with an in-memory term dictionary."""

    matrix = []
    for index, item in enumerate(terms):
        try:
            term = str(item["term"])
            weight = float(item["weight"])
        except KeyError as exc:
            raise ValueError(f"terms[{index}] must include 'term' and 'weight'") from exc
        matrix.append((term, weight))

    cleaned = legacy_utils.clean_documents([text])
    score, terms_found = legacy_score.weighted_score_calculation(" ".join(cleaned[0]), matrix)
    return {"score": score, "terms_found": terms_found}
