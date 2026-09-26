# ruff: noqa: E501,I001
"""Static contract tests for the Qwen3 semantic search and reranking workshop."""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

from qwen3_embedding_pipeline import samples

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb"
EMBED_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RERANK_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
RERANK_WEIGHT_SHA256 = "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def body():
    return "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])


def code_cells():
    return ["".join(cell["source"]) for cell in load()["cells"] if cell["cell_type"] == "code"]


def embedded_manifest(name):
    match = re.search(name + r' = json\.loads\(r"""(.*?)"""\)', body(), re.S)
    assert match, f"{name} not embedded"
    return json.loads(match.group(1))


def test_generator_parity():
    subprocess.run([sys.executable, str(REPO / "tools" / "build_semantic_search_reranking_workshop.py"), "--check"], cwd=REPO, check=True)


def test_metadata():
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "MULTI-CAPABILITY"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["worker_required"] is False
    assert meta["credentials_required"] is False
    assert meta["clean_runtime_evidence"] == "pending"
    assert [m["revision"] for m in meta["models"]] == [EMBED_REVISION, RERANK_REVISION]


def test_code_cells_compile():
    for cell in code_cells():
        compile(cell, "cell", "exec")


def test_embedding_manifest_matches_the_repository_snapshot():
    committed = json.loads((REPO / "weights" / "qwen3-embedding-0.6b" / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert embedded_manifest("EMBED_MANIFEST") == committed


def test_reranker_manifest_is_pinned():
    manifest = embedded_manifest("RERANK_MANIFEST")
    assert manifest["modelId"] == "Qwen/Qwen3-Reranker-0.6B"
    assert manifest["revision"] == RERANK_REVISION
    weights = next(f for f in manifest["files"] if f["path"] == "model.safetensors")
    assert weights["sha256"] == RERANK_WEIGHT_SHA256
    assert manifest["totalBytes"] == sum(f["bytes"] for f in manifest["files"])


def test_dataset_pins_match_the_repository():
    text = body()
    assert samples.CORPUS_BASE_URL in text.replace('"\n    "', "")
    for name, size, digest in samples.CORPUS_FILES.values():
        assert f'("{name}", {size:_}, "{digest}")' in text


def test_contract_present():
    text = body()
    for literal in [
        "USE_BYOD = False",
        "RERANK_K = 6",
        "SAMPLE_SEED = 42",
        "RUN_K_SWEEP = False",
        "trust_remote_code=False",
        "local_files_only=True",
        "YES_TOKEN_ID, NO_TOKEN_ID = 9693, 2152",
        "shortlist_coverage@k",
        "unrecoverable",
        "not-measurable",
        "document_embeddings.npz",
        "retrieval_results.csv",
        "reranked_results.csv",
        "metrics.json",
        "provenance.json",
    ]:
        assert literal in text


def test_no_runtime_repo_dependency():
    text = body()
    for forbidden in ["git clone ", "pip install -e", "raw.githubusercontent.com/kurtvalcorza", "import qwen3_embedding_pipeline", "import qwen3_reranker_pipeline", "dimer-backend"]:
        assert forbidden not in text


def test_clean_notebook():
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_stale_import_guard_tells_colab_users_to_restart_not_delete():
    # "Start a fresh runtime" read as Disconnect-and-delete on Colab, which discards the pins and repeats the error.
    text = "\n".join("".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
    assert "Restart session" in text
    assert "Do not disconnect or delete the runtime" in text
    assert "Start a fresh runtime" not in text


def test_install_keeps_a_numpy_the_kernel_already_loaded():
    # Colab imports NumPy at startup; reinstalling it left 2.1.3 in memory over 2.5.3 on disk, which broke later
    # imports (Notebook Spec RUN10 forbids a manual restart). The cell keeps a loaded NumPy 2.x and fails closed
    # if any module it depends on was replaced underneath the kernel.
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    install = next(cell for cell in cells if "pip" in cell and "install" in cell)
    assert 'NUMPY_PRELOADED' in install and '"numpy" in sys.modules' in install
    assert "if stale:" in install and "Restart session" in install
