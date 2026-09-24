"""Static contract tests for the semantic-search/reranking workshop."""
# ruff: noqa: E501
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
NOTEBOOK=REPO/"tutorials"/"DIMER_Semantic_Search_Reranking_Workshop.ipynb"

def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))

def body():
    return "\n".join("".join(cell.get("source",[])) for cell in load()["cells"])

def test_generator_parity():
    subprocess.run([sys.executable,str(REPO/"tools"/"build_semantic_search_reranking_workshop.py"),"--check"],cwd=REPO,check=True)

def test_metadata():
    meta=load()["metadata"]["dimer"]
    assert meta["notebook_spec"]=="2.1"
    assert meta["notebook_profile"]=="TASK-INFERENCE"
    assert meta["notebook_mode"]=="WORKSHOP"
    assert meta["standalone"] is True
    assert meta["clean_runtime_evidence"]=="pending"

def test_two_stage_contract():
    text=body()
    for literal in [
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Reranker-0.6B",
        'USE_BYOD = False  # @param',
        'BYOD_ZIP_PATH = ""  # @param',
        "candidate_recall@k",
        "conditional_reranker_top1",
        "semantic_search_per_query.csv",
        "semantic_search_provenance.json",
    ]:
        assert literal in text

def test_pinned_embedded_sources_no_runtime_clone():
    text=body()
    assert "115cf17fb35048dcadc187ac7dd36b28d99f9aab" in text
    assert "f13a58e65a7ee54343e8fa262c166308699c4f11" in text
    for forbidden in ["git clone ","pip install -e","dimer-backend"]:
        assert forbidden not in text

def test_clean_notebook():
    for cell in load()["cells"]:
        if cell["cell_type"]=="code":
            assert cell["execution_count"] is None
            assert cell["outputs"]==[]
