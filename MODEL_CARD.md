---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: feature-extraction
task: "Others - Text Embedding"
base_model: Qwen/Qwen3-Embedding-0.6B
date_published: "2025-06-03"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B)"
---

# Qwen3-Embedding-0.6B — Text Embedding Model (Encoder)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--Embedding--0.6B-ffcc4d?style=flat)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-QwenLM%2FQwen3--Embedding-181717?style=flat&logo=github&logoColor=white)](https://github.com/QwenLM/Qwen3-Embedding)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2506.05176-b31b1b.svg)](https://arxiv.org/abs/2506.05176)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/qwen3-embedding-pipeline/blob/main/tutorials/qwen3_embedding_colab.ipynb) [`qwen3_embedding_colab.ipynb`](https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/tutorials/qwen3_embedding_colab.ipynb)  
  *Text embeddings with the pinned `Qwen/Qwen3-Embedding-0.6B` weights: 1024-d last-token-pooled, L2-normalised vectors with instruction-aware queries; representations, not predictions; query–document cosine as a qualitative check only, no metric.*

---

#### Description

`Qwen/Qwen3-Embedding-0.6B` is the smallest of the Qwen3 Embedding series (Zhang et al., arXiv:2506.05176), pinned here to revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. The upstream README states it is built on `Qwen/Qwen3-0.6B-Base`; the pinned `config.json` is a `Qwen3ForCausalLM` decoder with 28 layers, hidden size 1024, 16 attention heads over 8 key-value heads, and a 32,768-token position budget. Used as an embedder, the decoder runs one forward pass over the text and the hidden state at the last token position is taken as the sentence vector (`1_Pooling/config.json`: `pooling_mode_lasttoken: true`), then L2-normalised; queries are prefixed with a task instruction, documents are not. Nothing is trained, fine-tuned, or conditioned in this repository. What it adds is the `Qwen3EmbeddingPipeline` class in `src/qwen3_embedding_pipeline/pipeline.py`: manifest verification (`verify_snapshot`), fresh-clone staging (`stage_missing_files`), a loader that refuses remote code and mutable caches, input validation with named ceilings, the README's exact prefix/pooling/normalisation contract, and provenance fields in every result.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is text embedding: input is a list of up to `MAX_BATCH = 64` strings with a `kind` of `"query"` or `"document"`; output is one 1024-dimensional unit-norm float32 vector per string, so that the dot product of a query vector and a document vector is their cosine similarity. Envisioned applications are dense retrieval over an operator's own corpus (documents embedded once, queries at request time), semantic deduplication and clustering of text collections, and use as a fixed feature extractor feeding a downstream classifier the operator trains; the upstream README also names code retrieval and bitext mining across the 100+ languages it claims. In a larger system the pipeline is the encoding component of a retrieval stack, paired with a vector index and, where precision matters, the sibling `qwen3-reranker-pipeline` on the top candidates.

###### Primary Intended Users

Intended users are machine-learning engineers, search and data engineers, and application developers building retrieval or clustering features for research prototypes, internal enterprise search, or the DIMER model workbench. The pipeline assumes its users know that an embedding is a representation with no label attached, that cosine similarity between two vectors is only meaningful relative to other pairs from the same model, that queries and documents must be embedded with the matching `kind` (mixing them silently degrades retrieval), that text beyond 8,192 tokens is cut off, and that retrieval quality on their own corpus must be measured with their own relevance judgements before deployment.

###### Out-of-scope use cases

1. **Capability boundary:** the model does not generate text, answer questions, classify into a fixed label set, or score a query-document pair directly; pair scoring is the sibling `qwen3-reranker-pipeline`, and text generation belongs to the language-model pipelines. It emits no token-level or span-level output.
2. **Input boundary:** `embed()` takes a list of 1 to 64 non-empty strings (`MAX_BATCH`; a bare string raises `TypeError`, an empty list or empty item raises `ValueError`); any item above 100,000 characters (`MAX_TEXT_CHARS`) is rejected before tokenisation, and any item that tokenises beyond 8,192 tokens (`MAX_TEXT_TOKENS`) is truncated and flagged `truncated: True`. Only `kind` values `"query"` and `"document"` are accepted. Images, audio, and structured records are not inputs.
3. **Decision boundary:** not for autonomous decisions about people from similarity scores — candidate screening, content moderation, fraud flags, or eligibility — without a human reviewing the retrieved items; a similarity score carries no calibrated meaning and the model's training data is not disclosed.

#### Factors

###### Groups

The pipeline is not human-centric by design — it maps arbitrary text to vectors — but the text it embeds routinely describes or is written by people, and the upstream pretraining and contrastive fine-tuning corpora are not enumerated or group-audited by the Qwen team in the pinned README. No group-level audit exists here either. Embedding bias transfers directly into the operator's retrieval pipeline: if names, dialects, or topics associated with a demographic group sit systematically closer to or farther from certain queries, ranking is skewed for that group. The downstream operator must run a fairness audit on their own retrieval task — for example, retrieval rate and rank position for otherwise-equivalent queries and documents that differ only in a name, gender marker, or language variety — before relying on the ranking.

###### Instrumentation

There is no physical sensor: the training data is text, produced by keyboards, web crawls, transcription, and machine translation, then tokenised by the Qwen2 byte-pair tokenizer (`tokenizer.json`, 151,669-entry vocabulary). The characteristics that materially affect the data are encoding, normalisation, and language mix at collection time, none of which the upstream README describes beyond the "100+ languages" claim and the note that most training instructions were written in English. At inference the pipeline consumes strings exactly as the caller passes them: no case folding, whitespace normalisation, HTML stripping, or language detection. Defects in the operator's upstream text pipeline — mis-decoded bytes, boilerplate, concatenated records, wrong-language text — reach the model unchanged, and the pipeline can detect only the shape errors it validates (type, emptiness, length ceilings) and the truncation it reports.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0` (the venv build is `2.14.0+cu130`), `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3` (`pyproject.toml`). The loader runs float32 on CPU and bfloat16 on CUDA (the checkpoint's native `torch_dtype`); the recorded smoke ran on CPU only (`CUDA_VISIBLE_DEVICES=-1`, `device="cpu"`), loading 1.19 GB of weights in 6.63 s and embedding four short texts in 0.44 s on the host described under "Runtime". Flash attention and the CUDA path were not exercised. Data environment: the model assumes natural-language or code text resembling its undisclosed web-scale training mix, with queries carrying an English task instruction as the README recommends; retrieval quality degrades, without any error, on domains far from that mix (specialised jargon, tables serialised as text, very short or very long inputs near the 8,192-token cut) and on languages with thin coverage.

#### Metrics

###### Performance Measures

The pipeline reports no performance measure. `embed()` returns representations, not predictions, and there is no ground truth in a batch of strings against which to score them; the only computed quantity is `cosine_similarity()`, a helper that returns the pairwise similarity matrix and is not a metric. Upstream reports MTEB scores for this checkpoint (for example a multilingual mean-task score of 64.33 in the pinned README's evaluation table); those are upstream-reported and were not reproduced by this pipeline. An operator who needs a measure must supply queries, a document corpus, and graded relevance judgements, then compute a ranking metric such as nDCG@10 or Recall@k over the cosine-ranked results — ranking metrics because retrieval is the intended use and a representation has no discrete correctness to count. The public `evaluation_report()` helper makes that absence machine-readable rather than silent: it always returns a report whose `verdict` is `not-measurable` with an empty `metrics` list, and whose `needs` field names the labelled retrieval or classification data that would be required to score this model.

###### Decision thresholds

No decision threshold is shipped, implicit or explicit: `embed()` applies no argmax, no cutoff, and no nearest-neighbour rule, and `cosine_similarity()` returns raw values in [-1, 1]. A threshold was withheld because the meaningful cutoff depends on the corpus, the query distribution, and the cost of a missed versus a spurious retrieval, none of which this repository can see. The operator owns it: choose a similarity cutoff or a top-k from precision-recall curves on labelled query-document pairs from the deployment, lowering it where a missed relevant document costs more than reviewing extra candidates and raising it where noise is expensive. The smoke run's values (0.7646 for a matching pair, 0.1414 for a mismatched one) are two observations reproduced from the upstream README, not calibration points.

###### Approaches to uncertainty and variability

No metric is reported, so no estimation procedure or dispersion applies; the operator who computes nDCG or recall on their own judgements owns the split design and any bootstrap or repeated-run estimate. Within the pipeline the forward pass is deterministic on a fixed device and dtype — there is no sampling, dropout is inactive under `torch.inference_mode`, and no seed is needed — but bfloat16 on CUDA versus float32 on CPU, kernel selection, and batch composition (left padding changes nothing in the pooled token but can shift low-order bits) move vector components in the third or fourth decimal place; the CPU smoke reproduced the README's four cosine values to four decimals. A cosine similarity is not a probability and is not calibrated; a caller who needs a confidence must fit a mapping from similarity to relevance on their own labelled pairs.

#### Ethical considerations and biases

###### Data

The upstream README discloses only that the model is built on `Qwen3-0.6B-Base`, supports 100+ languages, and that its training instructions were mostly English; the technical report (arXiv:2506.05176) describes large-scale synthetic and public retrieval pairs for contrastive training, and the base model's pretraining corpus is web-scale text that Qwen does not enumerate. Whether personal, copyrighted, or otherwise restricted text is included is therefore unknown, not ruled out. This repository distributes code, tests, and documentation; it does not vendor the weights in Git (`weights/qwen3-embedding-0.6b/model.safetensors` is git-ignored and reproduced from the pinned revision via `stage_missing_files`), and it ships no text corpus. The operator must audit the text they embed for personal, confidential, or legally restricted content and must treat the resulting vectors as derived personal data where the source text was; the pipeline performs no such check.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by anyone; the only validation performed is the offline unit suite and one CPU smoke recorded under "Runtime". Use in such domains is foreseeable — matching résumés to job descriptions, retrieving similar case files, finding related patient notes — and would be admissible only as a retrieval aid whose results a qualified person reviews, after an independent evaluation of retrieval quality and group-level behaviour on that deployment's own data, and with whatever regulatory clearance the domain requires.

###### Mitigations

1. **Supply-chain integrity:** `MODEL_ID` and the 40-hex `MODEL_REVISION` are module constants; `verify_snapshot()` checks `modelId`, `revision`, and the byte size and SHA-256 of all 11 manifest entries before any load, raising on the first mismatch (a test flips one hex digit and asserts it raises). `stage_missing_files()` fetches only manifest-listed files absent on disk, only at `MODEL_REVISION`, only when `allow_download=True`, and refuses a manifest naming another model. The loader passes `local_files_only=True` for the verified directory and `trust_remote_code=False` always.
2. **Input integrity:** `embed()` rejects a bare string, an empty or oversize batch, non-string or blank items, items above `MAX_TEXT_CHARS`, an unknown `kind`, and an empty instruction before the model runs (one test per check), and checks the backend's output shape after. The public `validate_inputs()` helper applies those same checks through the same private `_check_inputs()` function and returns an input manifest (schema, ceilings, per-input observations, verdict, findings), so a caller can record exactly what was accepted or rejected without duplicating the validation logic.
3. **Statistical mitigations:** none are implemented; the pipeline does no training, so there is no balancing or subsampling to apply.
4. **Reproducibility:** every runtime dependency is pinned with `==`; each result carries `model_id`, `model_revision`, `pooling`, `normalized`, `kind`, `instruction`, `n_tokens`, and `truncated`.
5. **Refusals:** no similarity threshold, nearest-neighbour rule, or index is exposed; the pipeline returns vectors and leaves ranking to the caller. Matryoshka truncation to fewer than 1024 dimensions, which upstream supports, is deliberately not exposed so every vector this package emits has one fixed `dim`.

###### Risks and harms

1. **Silent truncation:** text past 8,192 tokens is dropped, so a long document is represented by its head only; the `truncated` flag is set, and an operator who ignores it retrieves on partial content. Likely for long reports and transcripts.
2. **Kind mismatch:** embedding queries as documents (or vice versa) removes the instruction prefix the model was trained with and degrades ranking without any error; the operator bears the harm as silently worse search.
3. **Bias amplification in retrieval:** systematic proximity of names, dialects, or topics to certain queries skews who or what is retrieved; the data subject bears the harm, and it compounds when the ranking feeds a human decision.
4. **Out-of-distribution confidence:** unfamiliar domains or languages still yield high-looking similarities; the operator misreads noise as relevance.
5. **Privacy and leakage:** vectors are derived from the embedded text and can leak its content through nearest-neighbour or inversion attacks on the index; an operator who treats an index as non-personal data exposes the data subjects. Magnitude ranges from degraded search to a wrongful screening outcome when the decision boundary above is ignored.

###### Use cases

The pipeline must not be used to profile individuals from the text they write, to infer protected characteristics from embeddings, for surveillance or social scoring, or for unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access — including ranking people by similarity to a prototype in any of those settings. It must not be used to de-anonymise authors, to build indexes over text obtained without authorisation, or in any way that breaches the Apache-2.0 terms of the upstream weights or the DIMER deployment terms. These prohibitions hold even where the model would produce a plausible ranking.

## Immutable provenance

- Model: `Qwen/Qwen3-Embedding-0.6B`
- Revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- Manifest: `weights/qwen3-embedding-0.6b/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 11 files, `totalBytes` 1207487471
- `model.safetensors` (1,191,586,416 bytes) SHA-256: `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`
- `config.json` (727 bytes) SHA-256: `b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9`
- `1_Pooling/config.json` (313 bytes) SHA-256: `37bf193fa101f19101bfad9c31d3eb0f786e247b7b1e5cb7f007d730eed1ddbd`
- Upstream reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

## Input/output contract

- `Qwen3EmbeddingPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`: stages missing manifest files (only with `allow_download=True`), verifies the snapshot, loads `AutoTokenizer` (`padding_side="left"`) and `AutoModel` with `local_files_only=True`; `device` defaults to `cuda:0` when visible, else `cpu`.
- `embed(texts: list[str], kind: str = "document", instruction: str = DEFAULT_QUERY_INSTRUCTION) -> dict` with keys `embeddings` (list of 1024 float lists, unit norm), `dim` (1024), `pooling` (`"last_token"`), `normalized` (`True`), `kind`, `instruction` (the prefix used for queries, `None` for documents), `n_tokens` (per text), `truncated` (per text), `model_id`, `model_revision`.
- `cosine_similarity(a, b) -> list[list[float]]`: pairwise cosine matrix; a helper, not a metric.
- Constants: `EMBEDDING_DIM = 1024`, `MAX_TEXT_TOKENS = 8192`, `MAX_TEXT_CHARS = 100000`, `MAX_BATCH = 64`, `DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"`.
- No metric helper is shipped; retrieval evaluation needs relevance judgements the repository does not have.

## Runtime

- Pins (`pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`; dev `pytest==8.4.2`, `ruff==0.16.6`. Python 3.12, Windows venv `dimer-next16`.
- Executed 2026-09-12: `CUDA_VISIBLE_DEVICES=-1 python -m pytest -q -o addopts= tests` — 19 passed, exit 0; `ruff check src tests` clean.
- Smoke, executed on CPU (`CUDA_VISIBLE_DEVICES=-1`, `from_pretrained(device="cpu")`, float32): two README queries embedded with `kind="query"` and the two README documents with `kind="document"`; load 6.63 s, embedding 0.437 s, 7.06 s total; `dim` 1024, `n_tokens` [27, 23, 8, 31], no truncation, vector norm 1.0; cosine matrix [[0.7646, 0.1414], [0.1355, 0.6000]], equal to the upstream README's printed values to four decimals.
- Not executed: the CUDA/bfloat16 path, the `allow_download=True` Hub path, inputs near the 8,192-token ceiling against the real model, and any labelled retrieval evaluation.

## References

- Zhang, Y. et al. (2025). Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. https://arxiv.org/abs/2506.05176
- Upstream model card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (pinned README, revision above)
- Upstream code: https://github.com/QwenLM/Qwen3-Embedding
- Sibling reranker: https://github.com/kurtvalcorza/qwen3-reranker-pipeline
