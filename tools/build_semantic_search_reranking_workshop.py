# ruff: noqa: E501,I001
"""Generate the DIMER semantic-search and reranking workshop notebook."""
from __future__ import annotations
import argparse
import ast
import io
import json
import tokenize
from pathlib import Path
from semantic_search_reranking_workshop_source import CELLS
from semantic_search_isolated_runtime import LOCK_FILE, MANAGED_PYTHON, MAX_PIECE, SETUP_CELL_INDEX, SETUP_CELL_PLACEHOLDER, UV, pins, setup_cell

NOTEBOOK_NAME="DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb"
# No notebook cell line may exceed this many characters (long lines break hosted editors and diff review).
MAX_CELL_LINE=2000

def cell_sources():
    """CELLS with the generated isolated-environment setup cell substituted for its placeholder."""
    if CELLS[SETUP_CELL_INDEX]["source"]!=SETUP_CELL_PLACEHOLDER:
        raise SystemExit(f"CELLS[{SETUP_CELL_INDEX}] must be the setup-cell placeholder")
    sources=[cell["source"] for cell in CELLS]
    sources[SETUP_CELL_INDEX]=setup_cell()
    for index,source in enumerate(sources):
        if SETUP_CELL_PLACEHOLDER.strip() in source and index!=SETUP_CELL_INDEX:
            raise SystemExit(f"cell {index} carries the setup-cell placeholder")
        longest=max((len(line) for line in source.splitlines()),default=0)
        if longest>MAX_CELL_LINE:
            raise SystemExit(f"cell {index} has a {longest}-character line (limit {MAX_CELL_LINE}); split the literal into pieces")
    setup=ast.parse(sources[SETUP_CELL_INDEX])
    for node in setup.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="UV_URL":
            # Implicitly concatenated pieces parse to one Constant, so measure each string token instead.
            segment=ast.get_source_segment(sources[SETUP_CELL_INDEX],node.value)
            tokens=tokenize.generate_tokens(io.StringIO(segment).readline)
            pieces=[ast.literal_eval(t.string) for t in tokens if t.type==tokenize.STRING]
            if ast.literal_eval(node.value)!=UV["url"] or any(len(p)>MAX_PIECE for p in pieces):
                raise SystemExit("UV_URL pieces do not reassemble the pinned uv wheel URL")
    return sources

def build_notebook():
    rendered=[]
    for index,(cell,source) in enumerate(zip(CELLS,cell_sources(),strict=True)):
        base={"id":f"dimer-search-workshop-{index:02d}","metadata":cell.get("metadata", {}),"source":source.splitlines(keepends=True)}
        if cell["kind"]=="markdown":
            rendered.append({"cell_type":"markdown",**base})
        else:
            rendered.append({"cell_type":"code","execution_count":None,"outputs":[],**base})
    return {
        "cells":rendered,
        "metadata":{
            "accelerator":"GPU",
            "colab":{"gpuType":"T4","provenance":[]},
            "dimer":{
                "notebook_profile":"MULTI-CAPABILITY",
                "notebook_mode":"WORKSHOP",
                "notebook_spec":"2.1",
                "standalone":True,
                "capability":"two-stage semantic search and reranking",
                "carrier":"standalone Qwen3 embedding retrieval and cross-encoder reranking workshop",
                "models":[
                    {"id":"Qwen/Qwen3-Embedding-0.6B","revision":"97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"},
                    {"id":"Qwen/Qwen3-Reranker-0.6B","revision":"e61197ed45024b0ed8a2d74b80b4d909f1255473"},
                ],
                "dataset":"Banking77 (CC BY 4.0): 77 intent documents, 154 held-out test queries",
                "canonical_runtime":"NVIDIA Tesla T4",
                "worker_required":False,
                "credentials_required":False,
                "clean_runtime_evidence":"pending",
                "isolated_runtime":{
                    "mechanism":"uv isolated environment; every code cell after setup runs in one persistent worker",
                    "managed_python":MANAGED_PYTHON,
                    "uv":UV["version"],
                    "lock":f"tutorials/{LOCK_FILE.name}",
                    "pins":pins(),
                    "platform":"Linux x86_64 only",
                },
                "revisions":[
                    {"date":"2026-10-03","change":"moved to the uv isolated environment: no kernel install, no restart guard; hash-locked wheel-only requirements at unchanged pins; Linux x86_64 only","previous_blob":"99a726dcb8bd92f4a5ff0dd5fbc190b6a070629f"},
                ],
                "generated_from":{
                    "repository":"kurtvalcorza/qwen3-embedding-pipeline",
                    "source":"tools/semantic_search_reranking_workshop_source.py",
                    "generator":"tools/build_semantic_search_reranking_workshop.py",
                },
            },
            "kernelspec":{"display_name":"Python 3","name":"python3"},
            "language_info":{"name":"python"},
        },
        "nbformat":4,
        "nbformat_minor":5,
    }

def serialized():
    return json.dumps(build_notebook(),indent=1,ensure_ascii=False)+"\n"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=args.out or repo/"tutorials"/NOTEBOOK_NAME
    content=serialized()
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8")!=content:
            raise SystemExit(f"STALE: {out}; regenerate the workshop notebook")
        print(f"OK: {out}")
        return 0
    out.write_text(content,encoding="utf-8")
    print(out)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
