from __future__ import annotations

import csv
import json
from base64 import b64encode
from pathlib import Path

import pytest

from fumeus_mcp.adapters import (
    clean_text,
    generate_smoke_terms,
    score_records,
    score_text,
    upload_file,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_DATASET = ROOT_DIR / "samples" / "example_dataset.csv"
EXAMPLE_DICTIONARY = ROOT_DIR / "samples" / "example_dictionary.csv"


def read_example_records() -> list[dict[str, str]]:
    with open(EXAMPLE_DATASET, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_example_terms() -> list[dict[str, str]]:
    with open(EXAMPLE_DICTIONARY, "r", encoding="utf-8", newline="") as file:
        return [
            {"term": row["Term"], "weight": row["Weight"]}
            for row in csv.DictReader(file)
        ]


def test_generate_smoke_terms_returns_structured_terms():
    result = generate_smoke_terms(
        dataset_path=str(EXAMPLE_DATASET),
        header=True,
        text_column=0,
        label_column=1,
        ngram_length=1,
        mode="DRC",
        top_n=3,
    )

    assert result["count"] == 3
    assert result["mode"] == "DRC"
    assert {term["term"] for term in result["terms"]} == {"staff", "rude", "were"}
    assert {term["score"] for term in result["terms"]} == {2.82842712474619}


def test_upload_file_writes_text_to_upload_dir(tmp_path):
    result = upload_file(
        file_name="datasets/uploaded.csv",
        content="Record,Concern?\nStaff were rude,1\n",
        upload_dir=str(tmp_path),
    )

    uploaded = Path(result["path"])
    assert uploaded == tmp_path / "datasets" / "uploaded.csv"
    assert uploaded.read_text(encoding="utf-8") == "Record,Concern?\nStaff were rude,1\n"
    assert result == {
        "path": str(uploaded),
        "file_name": "datasets/uploaded.csv",
        "size_bytes": 34,
        "encoding": "text",
        "overwritten": False,
    }


def test_upload_file_decodes_base64_and_controls_overwrites(tmp_path):
    encoded = b64encode(b"term,weight\nrude,4\n").decode("ascii")

    result = upload_file("dictionary.csv", encoded, encoding="base64", upload_dir=str(tmp_path))
    uploaded = Path(result["path"])

    assert uploaded.read_bytes() == b"term,weight\nrude,4\n"
    assert result["size_bytes"] == 19
    assert result["encoding"] == "base64"

    with pytest.raises(FileExistsError, match="Upload already exists"):
        upload_file("dictionary.csv", encoded, encoding="base64", upload_dir=str(tmp_path))

    overwritten = upload_file(
        "dictionary.csv",
        "term,weight\nstaff,4\n",
        overwrite=True,
        upload_dir=str(tmp_path),
    )
    assert overwritten["overwritten"] is True
    assert uploaded.read_text(encoding="utf-8") == "term,weight\nstaff,4\n"


def test_upload_file_rejects_unsafe_names_and_invalid_encoding(tmp_path):
    with pytest.raises(ValueError, match="safe relative path"):
        upload_file("../escape.csv", "x", upload_dir=str(tmp_path))

    with pytest.raises(ValueError, match="encoding"):
        upload_file("safe.csv", "x", encoding="binary", upload_dir=str(tmp_path))

    with pytest.raises(ValueError, match="valid base64"):
        upload_file("safe.csv", "not base64", encoding="base64", upload_dir=str(tmp_path))


def test_generate_smoke_terms_can_keep_output_file(tmp_path):
    output = tmp_path / "terms.json"

    result = generate_smoke_terms(
        dataset_path=str(EXAMPLE_DATASET),
        header=True,
        text_column=0,
        label_column=1,
        output_path=str(output),
        top_n=2,
    )

    assert result["output_path"] == str(output)
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == result["terms"]


def test_generate_smoke_terms_validates_output_path(tmp_path):
    output = tmp_path / "missing" / "terms.json"

    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        generate_smoke_terms(
            dataset_path=str(EXAMPLE_DATASET),
            header=True,
            text_column=0,
            label_column=1,
            output_path=str(output),
        )


def test_score_records_returns_structured_records():
    result = score_records(
        dataset_path=str(EXAMPLE_DATASET),
        terms_path=str(EXAMPLE_DICTIONARY),
        header=True,
        terms_header=True,
    )

    assert result["count"] == 4
    assert result["records"][0] == {
        "text": "Service was unacceptable. The staff were rude!",
        "score": 12.618802154,
        "terms_found": ["rude", "staff", "unacceptable", "service"],
    }
    assert result["records"][1] == {
        "text": "Staff were so rude to me.",
        "score": 8.0,
        "terms_found": ["rude", "staff"],
    }


def test_clean_text_and_score_text():
    records = read_example_records()
    cleaned = clean_text([records[1]["Record"]])
    scored = score_text(records[1]["Record"], read_example_terms())

    assert cleaned == {
        "documents": [["service", "was", "unacceptable", "the", "staff", "were", "rude"]],
        "count": 1,
    }
    assert scored == {
        "score": 12.618802154,
        "terms_found": ["rude", "staff", "unacceptable", "service"],
    }


def test_generate_smoke_terms_validates_mode():
    with pytest.raises(ValueError, match="mode"):
        generate_smoke_terms(dataset_path=str(EXAMPLE_DATASET), mode="bad")
