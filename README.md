# Qwen3 Embedding Pipeline

DIMER-oriented inference wrapper for **Qwen3-Embedding-0.6B**, pinned to an immutable Hugging Face revision. The repository exposes 1024-dimensional, L2-normalised text embeddings with the upstream README's contract (left padding, last-token pooling, instruction prefix on queries only), a supply-chain check of the local weight snapshot, and machine-readable provenance.

## Upstream alignment

- Model: `Qwen/Qwen3-Embedding-0.6B`
- Revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- Upstream weight license: Apache-2.0
- Upstream task: text embedding / feature extraction (dense retrieval, clustering, similarity)
- Repository adaptation: **none**; inference only

## Quick start

```python
from qwen3_embedding_pipeline import Qwen3EmbeddingPipeline, cosine_similarity

pipe = Qwen3EmbeddingPipeline.from_pretrained()          # verifies weights/qwen3-embedding-0.6b first
queries = pipe.embed(["What is the capital of China?"], kind="query")
docs = pipe.embed(["The capital of China is Beijing.", "Gravity is a force."])
print(cosine_similarity(queries["embeddings"], docs["embeddings"]))   # [[~0.76, ~0.14]]
```

`embed()` takes 1..64 strings (`MAX_BATCH`). `kind="query"` prepends `Instruct: <instruction>\nQuery:` (default instruction from the upstream README; override with `instruction=`); `kind="document"` adds nothing. Text beyond 8,192 tokens (`MAX_TEXT_TOKENS`) is truncated and flagged per item in `truncated`; items above 100,000 characters (`MAX_TEXT_CHARS`) are rejected. Every result carries `dim`, `pooling`, `normalized`, `n_tokens`, `model_id` and `model_revision`.

## Weights layout

```
weights/qwen3-embedding-0.6b/
  config.json  1_Pooling/config.json  model.safetensors  tokenizer.json  tokenizer_config.json
  vocab.json  merges.txt  generation_config.json  modules.json  config_sentence_transformers.json
  README.md  dimer-base-manifest.json
```

`from_pretrained()` calls `stage_missing_files()` (fetches absent manifest entries at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (size + SHA-256 of every entry) and loads with `local_files_only=True`. Without a manifest it raises unless `allow_download=True`. See `docs/WEIGHTS.md`.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner and temporary manifests, never the weights.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/qwen3-embedding-pipeline/blob/main/tutorials/qwen3_embedding_colab.ipynb)

`tutorials/qwen3_embedding_colab.ipynb` is declared `TASK-INFERENCE` under DIMER Notebook Specification 1.0. Its default path embeds the four sentences of the pinned upstream README (bundled as literals with ids `q1`, `q2`, `d1`, `d2`; no download), surfaces the batch/token/character ceilings and the query instruction, resolves the pinned model through the package's staging and verification path, embeds queries and documents through `Qwen3EmbeddingPipeline.embed`, states that embeddings are representations with no intrinsic metric, shows the query–document cosine table as a qualitative check only, and exports identifiers alongside vectors (CSV) plus JSON provenance. CPU is slow for corpora but fine for a handful of sentences. BYOD is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks do not constitute clean-runtime notebook evidence. The clean-runtime run of the tutorial is pending; complete `docs/release-verification.md` against the exact release revision before calling the notebook release-grade.

## Licensing

This repository's code is Apache-2.0 (`LICENSE`). The packaged upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
