import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from qwen3_embedding_pipeline import (
    DEFAULT_QUERY_INSTRUCTION,
    DEFAULT_WEIGHTS_DIR,
    EMBEDDING_DIM,
    MAX_BATCH,
    MAX_TEXT_CHARS,
    MAX_TEXT_TOKENS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    Qwen3EmbeddingPipeline,
    cosine_similarity,
    stage_missing_files,
    verify_snapshot,
)
from qwen3_embedding_pipeline.pipeline import format_query

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"


def _fake_pipeline(calls=None, n_tokens=None):
    def runner(texts):
        if calls is not None:
            calls.append(list(texts))
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for i in range(len(texts)):
            vectors[i, i % EMBEDDING_DIM] = 3.0  # un-normalised on purpose
        return vectors, n_tokens or [4] * len(texts)

    return Qwen3EmbeddingPipeline(runner, "cpu")


def test_identity_constants_are_40_hex_and_match_manifest():
    assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REVISION)
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    if MANIFEST.is_file():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["modelId"] == MODEL_ID
        assert manifest["revision"] == MODEL_REVISION


def _write_snapshot(tmp_path: Path, content: bytes, sha256: str, revision: str = MODEL_REVISION) -> Path:
    (tmp_path / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [{"path": "config.json", "bytes": len(content), "sha256": sha256}],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_verify_snapshot_accepts_matching_digest(tmp_path):
    content = b'{"a": 1}'
    root = _write_snapshot(tmp_path, content, hashlib.sha256(content).hexdigest())
    assert verify_snapshot(root)["revision"] == MODEL_REVISION


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"a": 1}'
    good = hashlib.sha256(content).hexdigest()
    bad = ("0" if good[0] != "0" else "1") + good[1:]
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(_write_snapshot(tmp_path, content, bad))


def test_verify_snapshot_rejects_wrong_revision_and_size(tmp_path):
    content = b"xyz"
    digest = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(_write_snapshot(tmp_path, content, digest, revision="f" * 40))
    root = _write_snapshot(tmp_path, content, digest)
    (tmp_path / "config.json").write_bytes(b"xyzw")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)


def test_from_pretrained_refuses_without_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError):
        Qwen3EmbeddingPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


@pytest.mark.parametrize(
    ("texts", "kwargs", "exc"),
    [
        ("a single string", {}, TypeError),
        ([], {}, ValueError),
        (["x"] * (MAX_BATCH + 1), {}, ValueError),
        (["ok", 42], {}, TypeError),
        (["ok", "   "], {}, ValueError),
        (["a" * (MAX_TEXT_CHARS + 1)], {}, ValueError),
        (["ok"], {"kind": "passage"}, ValueError),
        (["ok"], {"kind": "query", "instruction": ""}, ValueError),
    ],
)
def test_embed_rejects_bad_input(texts, kwargs, exc):
    calls = []
    pipe = _fake_pipeline(calls=calls)
    with pytest.raises(exc):
        pipe.embed(texts, **kwargs)
    assert calls == []  # the model never ran


def test_embed_document_output_fields_and_normalisation():
    calls = []
    pipe = _fake_pipeline(calls=calls)
    result = pipe.embed(["The capital of China is Beijing.", "Explain gravity"])
    assert calls == [["The capital of China is Beijing.", "Explain gravity"]]  # no prefix on documents
    assert len(result["embeddings"]) == 2 and len(result["embeddings"][0]) == EMBEDDING_DIM
    assert abs(np.linalg.norm(result["embeddings"][0]) - 1.0) < 1e-6
    assert result["embeddings"][0][0] == 1.0  # 3.0 / 3.0 after L2 normalisation
    assert result["dim"] == EMBEDDING_DIM
    assert result["pooling"] == "last_token"
    assert result["normalized"] is True
    assert result["kind"] == "document"
    assert result["instruction"] is None
    assert result["n_tokens"] == [4, 4]
    assert result["truncated"] == [False, False]
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION


def test_embed_query_uses_instruction_prefix():
    calls = []
    pipe = _fake_pipeline(calls=calls)
    result = pipe.embed(["What is the capital of China?"], kind="query")
    expected = f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery:What is the capital of China?"
    assert calls == [[expected]]
    assert format_query("q", "do x") == "Instruct: do x\nQuery:q"
    assert result["instruction"] == DEFAULT_QUERY_INSTRUCTION
    custom = pipe.embed(["q"], kind="query", instruction="Retrieve code")
    assert calls[-1] == ["Instruct: Retrieve code\nQuery:q"]
    assert custom["instruction"] == "Retrieve code"


def test_embed_flags_truncation_at_token_ceiling():
    pipe = _fake_pipeline(n_tokens=[MAX_TEXT_TOKENS, 10])
    result = pipe.embed(["long", "short"])
    assert result["truncated"] == [True, False]


def test_cosine_similarity_matrix():
    sims = cosine_similarity([[1.0, 0.0], [0.0, 2.0]], [[3.0, 0.0]])
    assert abs(sims[0][0] - 1.0) < 1e-6 and abs(sims[1][0]) < 1e-6
    with pytest.raises(ValueError):
        cosine_similarity([[1.0, 0.0]], [[1.0, 0.0, 0.0]])


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)
