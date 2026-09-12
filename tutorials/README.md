# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/qwen3-embedding-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/qwen3-embedding-pipeline/blob/main/tutorials/qwen3_embedding_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--Embedding--0.6B-ffcc4d?style=flat)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
[![Upstream](https://img.shields.io/badge/Upstream-QwenLM%2FQwen3--Embedding-181717?style=flat&logo=github&logoColor=white)](https://github.com/QwenLM/Qwen3-Embedding)
[![arXiv](https://img.shields.io/badge/arXiv-2506.05176-b31b1b.svg)](https://arxiv.org/abs/2506.05176)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `qwen3_embedding_colab.ipynb` | `TASK-INFERENCE` | text embeddings (1024-d, last-token pooled, L2-normalised, instruction-aware queries) with `Qwen/Qwen3-Embedding-0.6B`; representations, not predictions; §20.6 obligations; cosine between queries and documents as a qualitative check only, no metric | CPU (CUDA used automatically when available); CPU is slow for corpora but fine for a handful of sentences (card-measured 6.6 s load, 0.44 s for four texts) | one UTF-8 text file (one document per line, ≤ 64 lines) plus a query typed into the form, gated off by default | **Candidate** — static checks pass; the clean-runtime execution run is pending and will be recorded in `../docs/release-verification.md`, which must be reviewed for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `Qwen3EmbeddingPipeline` from the repository public API rather than reimplementing model loading; model acquisition goes through the package: `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the manifest entries a fresh clone lacks, at the pinned revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads the verified files (`local_files_only=True`, `trust_remote_code=False`, float32 on CPU / bfloat16 on CUDA; the notebook never calls `huggingface_hub`).
- Embedding obligations (Notebook Specification §20.6): the notebook states the shape (one 1024-d vector per text), the pooling policy (last token, L2-normalised), the unit (per text; no per-token or chunked output), that missing data has no meaning (empty strings are rejected), that embeddings are representations rather than predictions, and it exports identifiers alongside vectors (`outputs/qwen3_embedding_vectors.csv`: `id`, `kind`, `n_tokens`, `truncated`, `e0000…e1023`).
- The default sample is the four sentences from the pinned upstream README (two queries, two documents; Apache-2.0), bundled as string literals with stable ids `q1`, `q2`, `d1`, `d2`; the cosine table is a qualitative check that the contract works, not a retrieval score (nDCG/recall need a labelled query–document set), and the card-recorded smoke matrix is quoted as one observation.
- Ceilings `MAX_BATCH` (64), `MAX_TEXT_TOKENS` (8,192), `MAX_TEXT_CHARS` (100,000), `EMBEDDING_DIM` (1024) are printed before the model runs; the query instruction is printed because it is part of the query vector; token-level truncation flags are surfaced after embedding.
- Precision is stated explicitly: float32 on CPU, bfloat16 on CUDA (values differ in the second or third decimal).
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
