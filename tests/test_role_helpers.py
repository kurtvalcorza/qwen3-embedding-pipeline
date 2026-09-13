"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest

from qwen3_embedding_pipeline import (
    DEFAULT_QUERY_INSTRUCTION,
    EMBEDDING_DIM,
    INPUT_SCHEMA,
    MAX_BATCH,
    MAX_TEXT_CHARS,
    MAX_TEXT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)


def _result(n: int = 2) -> dict:
    return {
        "embeddings": [[0.0] * EMBEDDING_DIM for _ in range(n)],
        "dim": EMBEDDING_DIM,
        "pooling": "last_token",
        "normalized": True,
        "kind": "document",
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(["alpha", "beta beta"], names=["d1", "d2"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["batch"] == [1, MAX_BATCH]
    assert manifest["schema"]["text_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["text_tokens"] == [1, MAX_TEXT_TOKENS]
    assert manifest["schema"]["embedding_dim"] == EMBEDDING_DIM
    assert manifest["inputs"] == [
        {"id": "d1", "chars": 5, "kind": "document"},
        {"id": "d2", "chars": 9, "kind": "document"},
    ]
    assert manifest["kind"] == "document"
    assert manifest["instruction"] is None
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_query_kind_records_the_instruction_and_default_ids() -> None:
    manifest = validate_inputs(["what is gravity?"], kind="query")
    assert manifest["kind"] == "query"
    assert manifest["instruction"] == DEFAULT_QUERY_INSTRUCTION
    assert [entry["id"] for entry in manifest["inputs"]] == ["text-0"]


def test_validate_inputs_rejects_like_embed() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs("a bare string")
    with pytest.raises(ValueError, match=f"1\\.\\.{MAX_BATCH}"):
        validate_inputs(["x"] * (MAX_BATCH + 1))
    with pytest.raises(ValueError, match="is empty"):
        validate_inputs(["  "])
    with pytest.raises(ValueError, match="ceiling is"):
        validate_inputs(["x" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(ValueError, match="kind must be one of"):
        validate_inputs(["x"], "passage")
    with pytest.raises(ValueError, match="instruction must be a non-empty str"):
        validate_inputs(["x"], "query", "")
    with pytest.raises(ValueError, match="names must have one entry per text"):
        validate_inputs(["x"], names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_texts"] == 2
    assert report["sample_kind"] == "synthetic"
    assert "nDCG@k or recall@k" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_stays_not_measurable_when_labels_are_supplied() -> None:
    report = evaluation_report(_result(1), [1], sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["sample_kind"] == "BYOD"
    assert "labels were supplied but no metric helper exists" in report["reason"]
