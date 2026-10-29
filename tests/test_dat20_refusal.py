"""DAT19/DAT20: a refused BYOD input renders exactly one `Refused: <file>: <reason>` line, with no traceback.

Executes the committed Section 4 BYOD block with the real loader (or stubs). Covers the scripted hosted
refusal input (`BYOD_PATH='/etc/hostname'`: a file without a dataset suffix), a directory, a CSV missing
columns, an empty path outside Colab, the in-cell invalidation, and that a genuine error keeps its own type.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from qwen3_embedding_pipeline import samples

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "qwen3_embedding_colab.ipynb"
REFUSAL_CELL = "qwen3_embedding-16"
BYOD_END = "else:\n    corpus = read_corpus("


def _byod_cell() -> dict:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    found = [c for c in cells if c["cell_type"] == "code" and "if USE_BYOD:" in "".join(c["source"])]
    assert len(found) == 1
    return found[0]


def _block() -> str:
    source = "".join(_byod_cell()["source"])
    return source[source.index("for _stale in") : source.index(BYOD_END)]


def _run(path: str, loader=samples.load_byod_dataset) -> dict:
    namespace = {
        "USE_BYOD": True,
        "BYOD_PATH": path,
        "Path": Path,
        "SPLIT_SEED": 42,
        "DatasetRefused": samples.DatasetRefused,
        "validate_dataset": samples.validate_dataset,
        "load_byod_dataset": loader,
        "split_dataset": samples.split_dataset,
        "splits": {"train": ["stale"]},
        "train_records": ["stale"],
        "adapt_result": {"stale": True},
    }
    try:
        exec(compile(_block(), "<byod>", "exec"), namespace)
    finally:
        _run.namespace = namespace
    return namespace


def _refused(path: str) -> str:
    with pytest.raises(samples.DatasetRefused) as caught:
        _run(path)
    rendered = caught.value._render_traceback_()
    assert caught.value.dimer_refusal is True
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    return rendered[0]


@pytest.fixture(autouse=True)
def _off_colab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)


def test_scripted_hosted_input_a_file_with_no_dataset_suffix(tmp_path):
    hostname = tmp_path / "hostname"
    hostname.write_text("colab-vm\n")
    line = _refused(str(hostname))
    assert line.startswith("Refused: hostname: BYOD datasets must be .csv, .json or .jsonl"), line


def test_a_directory_and_a_missing_path(tmp_path):
    assert _refused(str(tmp_path)).startswith(f"Refused: {tmp_path}: BYOD_PATH is not a file")
    assert _refused("nope.csv").startswith("Refused: nope.csv: BYOD_PATH is not a file")


def test_csv_missing_columns(tmp_path):
    housing = tmp_path / "california_housing_test.csv"
    housing.write_text("longitude,latitude\n1,2\n")
    line = _refused(str(housing))
    assert line.startswith("Refused: california_housing_test.csv: CSV is missing columns"), line


def test_empty_path_off_colab():
    assert _refused("").startswith("Refused: (no file): USE_BYOD is True but BYOD_PATH is empty")


def test_refusal_invalidates_earlier_results_in_the_cell():
    with pytest.raises(samples.DatasetRefused):
        _run("nope.csv")
    left = _run.namespace
    assert not {"splits", "train_records", "adapt_result"} & set(left)


def test_genuine_error_keeps_its_type(tmp_path):
    data = tmp_path / "pairs.csv"
    data.write_text("id,query,positive\n")

    def broken(path):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(str(data), loader=broken)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_refusal_cell_id():
    assert _byod_cell()["id"] == REFUSAL_CELL
