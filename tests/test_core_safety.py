from __future__ import annotations

import csv

import pytest

import generate
import score


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def test_csv_to_list_preserves_mixed_columns_and_adjusts_generate_text_column(tmp_path):
    dataset = tmp_path / "mixed.csv"
    output = tmp_path / "terms.json"
    write_csv(
        dataset,
        [
            ["id", "label", "narrative", "amount"],
            ["1", "1", "Smoke in cabin", "10.5"],
            ["2", "0", "Routine inspection", "9.0"],
        ],
    )

    x, y = generate.csv_to_list(str(dataset), header=True, y_column=1)

    assert x == [("1", "Smoke in cabin", "10.5"), ("2", "Routine inspection", "9.0")]
    assert y == [1.0, 0.0]

    generate.main(
        str(dataset),
        header=True,
        x_column=2,
        y_column=1,
        output_fname=str(output),
        format="json",
        top_n=2,
    )

    assert output.exists()


def test_csv_to_list_rejects_bad_y_values(tmp_path):
    dataset = tmp_path / "bad-label.csv"
    write_csv(dataset, [["text", "label"], ["Smoke", "yes"]])

    with pytest.raises(ValueError, match="non-numeric y value"):
        generate.csv_to_list(str(dataset), header=True, y_column=1)


def test_generate_rejects_invalid_mode_before_scoring():
    with pytest.raises(ValueError, match="mode must be one of"):
        generate.ir_score(["term"], ((("term",), 1),), [1], mode="BAD")


def test_main_functions_reject_invalid_output_format(tmp_path):
    dataset = tmp_path / "dataset.csv"
    terms = tmp_path / "terms.csv"
    write_csv(dataset, [["text", "label"], ["Smoke", "1"]])
    write_csv(terms, [["term", "weight"], ["smoke", "1"]])

    with pytest.raises(ValueError, match="format must be either"):
        generate.main(str(dataset), header=True, y_column=1, format="xml")

    with pytest.raises(ValueError, match="format must be either"):
        score.main(str(dataset), str(terms), str(tmp_path / "scores.xml"), header=True, header2=True, format="xml")


def test_output_writers_reject_missing_directories(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        generate.terms_to_json({"term": 1.0}, str(missing_dir / "terms.json"))

    with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
        score.scores_to_json(["text"], [[1.0, ["term"]]], str(missing_dir / "scores.json"))


def test_output_writers_reject_directory_paths(tmp_path):
    with pytest.raises(IsADirectoryError, match="Output path is a directory"):
        generate.terms_to_csv({"term": 1.0}, str(tmp_path))

    with pytest.raises(IsADirectoryError, match="Output path is a directory"):
        score.scores_to_csv(["text"], [[1.0, ["term"]]], str(tmp_path))
