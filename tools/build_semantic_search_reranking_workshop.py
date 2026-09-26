# ruff: noqa: E501,I001
"""Generate the DIMER semantic-search and reranking workshop notebook."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from semantic_search_reranking_workshop_source import CELLS

NOTEBOOK_NAME="DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb"

def build_notebook():
    rendered=[]
    for index,cell in enumerate(CELLS):
        base={"id":f"dimer-search-workshop-{index:02d}","metadata":cell.get("metadata", {}),"source":cell["source"].splitlines(keepends=True)}
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
