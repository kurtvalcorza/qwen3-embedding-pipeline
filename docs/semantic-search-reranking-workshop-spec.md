# DIMER Qwen3 Semantic Search and Reranking Workshop

**Proposed filename:** `DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb`  
**Notebook Specification:** DIMER `NOTEBOOK_SPEC.md` **v2.1**  
**Profile:** `MULTI-CAPABILITY`  
**Pedagogical mode:** `WORKSHOP`  
**Standalone:** `true`  
**Canonical workflow:** Frozen inference only; no fine-tuning or adapter production  
**Recommended runtime:** CUDA GPU / Tesla T4 or equivalent  
**CPU support:** Supported, but reranking will be substantially slower  
**Primary dataset:** Banking77, CC BY 4.0  
**Default evaluation:** 154 held-out queries × 77 candidate intent documents  
**Default rerank depth:** `RERANK_K = 6`

---

# 1. Purpose

This notebook demonstrates how two complementary DIMER models can be composed into a practical **two-stage semantic retrieval system**:

1. **Qwen3-Embedding-0.6B** performs efficient first-stage semantic retrieval over a document collection.
2. **Qwen3-Reranker-0.6B** performs more expensive cross-encoder scoring over a small shortlist returned by the retriever.

The notebook is not another model fine-tuning tutorial. Both individual pipeline repositories already provide release-grade E2E notebooks covering their respective adaptation workflows.

Instead, this workshop teaches the **application architecture**:

`documents → embeddings/index → query embedding → candidate retrieval → shortlist → cross-encoder reranking → final ranked results`

The principal learning objective is to show why semantic retrieval and reranking solve different parts of the search problem and why their composition is useful.

---

# 2. Learning objectives

By the end of the notebook, the learner should be able to:

1. distinguish a **bi-encoder / embedding retriever** from a **cross-encoder reranker**;
2. encode a document collection once and reuse those vectors for multiple queries;
3. retrieve candidates using cosine similarity between normalized embeddings;
4. pass only the top-*k* candidates to a cross-encoder reranker;
5. evaluate the retriever and the composed retrieval→reranking system separately;
6. explain why a reranker cannot recover a relevant document excluded by the first-stage retriever;
7. identify cases where reranking improves, preserves, or worsens the first-stage ordering;
8. understand the quality/compute tradeoff controlled by rerank depth;
9. export ranked search results and complete model/data provenance; and
10. replace the sample corpus with a user-provided document collection using the BYOD path.

---

# 3. Models

## 3.1 First-stage retriever

**DIMER profile:** Qwen3-Embedding-0.6B Text Embedding Model (Encoder)

| Field | Required value |
|---|---|
| Upstream model | `Qwen/Qwen3-Embedding-0.6B` |
| Immutable revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| License | Apache-2.0 |
| Weight format | `safetensors` |
| `model.safetensors` SHA-256 | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| Embedding dimension | 1024 |
| Pooling | Last-token pooling |
| Normalization | L2-normalized |
| Padding | Left |
| Maximum inference tokens | 8,192 |
| Maximum input characters | 100,000 |
| Maximum model call batch | 64 texts |
| Query semantics | Query receives instruction prefix |
| Document semantics | No query instruction prefix |

The workshop-specific query instruction is:

`Given a customer support message, retrieve the banking intent it expresses`

Embeddings MUST be treated as representations rather than predictions. Cosine similarity MUST NOT be described as a calibrated relevance probability.

---

## 3.2 Second-stage reranker

**DIMER profile:** Qwen3-Reranker-0.6B Text Reranking Model (Cross-Encoder)

| Field | Required value |
|---|---|
| Upstream model | `Qwen/Qwen3-Reranker-0.6B` |
| Immutable revision | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| License | Apache-2.0 |
| Weight format | `safetensors` |
| `model.safetensors` SHA-256 | `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b` |
| Scoring | `yes` vs `no` logit softmax |
| `YES_TOKEN_ID` | 9693 |
| `NO_TOKEN_ID` | 2152 |
| Maximum inference tokens | 8,192 total prompt |
| Maximum input characters | 100,000 per query/document |
| Maximum model call batch | 32 query-document pairs |
| Threshold | None |

Workshop instruction:

`Given a customer support message, judge whether the document names the banking intent it expresses`

The returned score is an **uncalibrated relevance score**, not a probability. The notebook MUST NOT apply or imply a universal decision threshold.

---

# 4. Why both models are required

The notebook MUST explicitly introduce the computational distinction.

## Embedding retrieval

Each document is independently encoded once:

`document → vector`

Each query is independently encoded:

`query → vector`

Candidate retrieval is then performed by vector similarity.

For a corpus of repeatedly searched documents, document embeddings can be precomputed and reused.

## Cross-encoder reranking

The reranker jointly processes:

`query + candidate document → relevance score`

This generally permits richer query-document interaction, but requires a model forward pass for every pair.

The resulting architecture is:

`large corpus`
→ **embedding retriever**
→ `small candidate set`
→ **cross-encoder reranker**
→ `final ranking`

The notebook MUST make the fundamental recall constraint explicit:

> The reranker can reorder only the documents it receives. If the relevant document is absent from the first-stage shortlist, the reranker cannot recover it.

This concept is central to the workshop.

---

# 5. Dataset contract

## 5.1 Default sample

Use the same digest-pinned Banking77 source already used by the two live DIMER carriers.

**Dataset:** Banking77  
**License:** CC BY 4.0  
**Source release:** `PolyAI-LDN/task-specific-datasets @ 57ec275d8078af65b7731c2a98be812d844a6d6b`

Required files:

| Split | File | Expected bytes | SHA-256 |
|---|---|---:|---|
| Train | `train.csv` | 839,073 | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| Test | `test.csv` | 239,961 | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

Expected source counts:

- train: 10,003 messages;
- test: 3,080 messages;
- intents: 77.

The notebook MUST verify byte size and SHA-256 before accepting either file.

---

## 5.2 Search corpus construction

Each of the 77 Banking77 intent identifiers becomes one candidate document.

Example:

`card_payment_wrong_exchange_rate`

becomes:

`card payment wrong exchange rate`

The default corpus therefore consists of:

**77 unique documents**

Each document MUST have:

```text
doc_id
text
intent
```

Example:

```text
intent_042
cash withdrawal
cash_withdrawal
```

This corpus is deliberately small and interpretable. It allows learners to inspect an entire retrieval problem without introducing an approximate nearest-neighbor system or vector database.

The notebook MUST state that intent phrases are a **tutorial retrieval surrogate**, not representative of the complexity of production document search.

---

## 5.3 Evaluation queries

Use a deterministic, balanced sample from the official Banking77 **test** partition.

Default:

**154 queries = 2 held-out messages × 77 intents**

Seed:

`SAMPLE_SEED = 42`

Each record:

```text
query_id
query
gold_doc_id
intent
```

No training, validation, adaptation, or model selection is performed in this notebook.

The sample is therefore an **evaluation/tutorial set**, not a training dataset.

The notebook MUST state that pretraining overlap with Banking77 cannot be ruled out.

---

# 6. Default form parameters

A configuration cell MUST expose top-level literals suitable for Colab forms and automated execution.

```python
USE_BYOD = False
DOCUMENTS_PATH = ""
QUERIES_PATH = ""

SAMPLE_SEED = 42
RERANK_K = 6

EMBEDDING_INSTRUCTION = "Given a customer support message, retrieve the banking intent it expresses"
RERANK_INSTRUCTION = "Given a customer support message, judge whether the document names the banking intent it expresses"

OUTPUT_DIR = "outputs/qwen3_semantic_search"
```

Optional workshop experiment:

```python
RUN_K_SWEEP = False
K_SWEEP_VALUES = [3, 6, 10]
```

The default `Run all` path MUST require no edits.

`RERANK_K = 6` is selected **a priori** because the existing DIMER reranker tutorial uses six-document shortlists. It MUST NOT be presented as a value optimized on the workshop test set.

---

# 7. Runtime and dependencies

The notebook SHOULD inherit the common tested runtime pins used by both live pipelines rather than introducing another dependency stack:

```text
torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0
transformers==4.57.6
huggingface-hub==0.36.2
safetensors==0.8.0
numpy==2.5.3
```

Python and all principal versions MUST be printed after installation.

The notebook MUST report:

```text
Python
PyTorch
Transformers
Hugging Face Hub
NumPy
device
CUDA availability
model dtype
```

GPU should be selected automatically when available.

Both checkpoints together account for roughly 2.4 GB of serialized model weights. Peak host/GPU memory MUST be measured during release qualification rather than inferred from checkpoint size.

No repository clone, DIMER service, API key, interactive authentication, external worker, or runtime download of DIMER source is permitted.

---

# 8. Standalone implementation

The notebook MUST contain a small standalone reference implementation for both inference contracts.

It MUST NOT:

- import `qwen3_embedding_pipeline`;
- import `qwen3_reranker_pipeline`;
- clone either repository;
- fetch Python source from GitHub at runtime; or
- call DIMER inference services.

The notebook generator MAY use the carrier repositories as build-time sources when constructing the notebook.

The generated notebook SHOULD embed:

- both immutable model identities;
- both snapshot manifests;
- shared digest-verification helpers;
- the embedding inference helper;
- the reranking inference helper;
- Banking77 acquisition/validation helpers;
- ranking metrics; and
- output/provenance helpers.

The notebook implementation MUST preserve the externally meaningful semantics of the live DIMER carriers.

---

# 9. Supply-chain behavior

For each model:

1. declare model ID and immutable revision;
2. include the corresponding verified manifest in the notebook;
3. download only manifest-listed files;
4. download from the immutable revision;
5. verify expected file size and SHA-256;
6. fail closed on any mismatch;
7. use `trust_remote_code=False`;
8. load only from the verified local snapshot; and
9. display the effective model ID and revision.

No mutable `main` revision is permitted.

A failed verification MUST NOT fall back to another checkpoint.

---

# 10. Canonical workflow

The primary `Run all` sequence is:

`Purpose`
→ `Runtime`
→ `Model provenance`
→ `Dataset provenance`
→ `Validate corpus/query contracts`
→ `Lexical/random baselines`
→ `Build document embedding index`
→ `Dense retrieval`
→ `Evaluate retriever`
→ `Build top-k shortlist`
→ `Cross-encoder reranking`
→ `Evaluate composed system`
→ `Failure analysis`
→ `Export`
→ `Interpretation`
→ `Optional BYOD / experiments`

---

# 11. Stage A — validation

Validation MUST occur before model execution.

Default checks:

### Corpus

- exactly 77 intent documents;
- unique `doc_id`;
- unique intent;
- non-empty text;
- no document exceeds `MAX_TEXT_CHARS`;
- document count ≤ 1,000;
- no duplicate normalized texts.

### Queries

- exactly 154 default queries;
- unique `query_id`;
- non-empty query;
- every `gold_doc_id` exists in the corpus;
- no query exceeds `MAX_TEXT_CHARS`;
- all 77 intents represented exactly twice.

The notebook MUST print a compact validation summary before loading models.

---

# 12. Stage B — non-neural baselines

Two context baselines SHOULD be reported.

## Random floor

For 77 candidates:

```text
expected recall@1 = 1 / 77
```

Other random ranking expectations MAY be calculated analytically.

## Lexical baseline

Use the same simple Jaccard/token-overlap semantics already present in the DIMER carrier utilities.

The baseline MUST rank all 77 intent documents for every query.

Report:

- recall@1;
- recall@3;
- recall@6;
- recall@10;
- MRR;
- median rank.

This establishes a no-neural-search comparison without introducing another search package.

---

# 13. Stage C — document indexing

Load `Qwen/Qwen3-Embedding-0.6B`.

Embed all 77 corpus documents using:

```text
kind = document
```

Requirements:

- batch automatically under `MAX_BATCH = 64`;
- no query instruction on documents;
- output vectors shape `(77, 1024)`;
- verify finite values;
- verify norm ≈ 1.0 within a documented tolerance;
- preserve document identifiers.

The notebook MUST explain that this is the **offline/indexing step**.

Measure and report:

```text
n_documents
embedding_dim
index_build_seconds
device
dtype
```

---

# 14. Stage D — semantic candidate retrieval

Embed all sample queries using:

```text
kind = query
instruction = EMBEDDING_INSTRUCTION
```

Compute cosine similarity using the dot product of normalized vectors.

For each query, produce the complete 77-document ranking.

Required retriever metrics:

- `recall@1`
- `recall@3`
- `recall@6`
- `recall@10`
- `mrr`
- `median_rank`

Also report:

```text
mean query embedding time
total retrieval time
number of queries
number of documents
```

These timings are tutorial observations for the measured runtime and MUST NOT be presented as general benchmarks.

---

# 15. Stage E — shortlist construction

For every query:

```python
shortlist = embedding_ranking[:RERANK_K]
```

Default:

`RERANK_K = 6`

For each shortlist record retain:

```text
query_id
query
candidate_doc_id
candidate_text
retrieval_rank
retrieval_score
is_gold
```

Calculate:

### `shortlist_coverage@K`

Fraction of queries for which the gold document is contained anywhere in the shortlist.

This is equivalent to retriever `recall@K`, but SHOULD be renamed in this section to emphasize the system interpretation.

The notebook MUST explicitly explain:

> `shortlist_coverage@K` is an upper bound on the number of queries the reranking stage can possibly solve correctly at rank 1.

---

# 16. Stage F — cross-encoder reranking

Unload the embedding model before loading the reranker **if runtime memory measurements show this materially improves portability**.

Otherwise both MAY remain resident.

Load:

`Qwen/Qwen3-Reranker-0.6B`

For each query, score its top-6 candidate pairs.

Pairs MUST be automatically chunked into groups of no more than:

`MAX_PAIRS = 32`

The reranker MUST use:

`RERANK_INSTRUCTION`

Each result retains:

```text
query_id
doc_id
retrieval_rank
retrieval_score
reranker_score
reranked_rank
is_gold
```

The notebook MUST state directly beside the output:

> The reranker score is an uncalibrated model relevance score. It is used only to order the supplied shortlist.

No threshold is applied.

---

# 17. End-to-end metrics

The notebook MUST distinguish **retrieval quality** from **reranking quality**.

## 17.1 Full-system metrics

Across all 154 queries:

- `pipeline_recall@1`
- `pipeline_recall@3`
- `pipeline_mrr@K`
- `shortlist_coverage@K`

If the gold document was absent from the shortlist:

- its end-to-end rank is treated as missing/outside `K`;
- reciprocal rank contribution is `0`.

This prevents the evaluation from hiding first-stage retrieval failures.

---

## 17.2 Conditional reranker metrics

For queries where the gold document **was present** in the shortlist:

- conditional reranker recall@1;
- conditional reranker MRR;
- median reranked gold rank.

These metrics isolate the behavior of the second stage from failures caused by candidate generation.

They MUST be clearly labelled **conditional** and MUST NOT replace end-to-end metrics.

---

## 17.3 Reranking effects

For each query classify the effect as:

### Helped
Gold rank improved after reranking.

### Hurt
Gold rank worsened after reranking.

### Unchanged
Gold rank did not change.

### Unrecoverable
Gold document was absent from the initial shortlist.

Report counts and percentages for all four groups.

This table is one of the primary instructional outputs.

---

# 18. Comparative result table

The notebook SHOULD produce one summary table shaped approximately as follows:

| System | Candidate pool | Recall@1 | Recall@3 | Recall@6 | MRR |
|---|---:|---:|---:|---:|---:|
| Random ranking | 77 | measured/analytic | — | — | analytic |
| Lexical baseline | 77 | measured | measured | measured | measured |
| Qwen3 Embedding | 77 | measured | measured | measured | measured |
| Qwen3 Embedding → Qwen3 Reranker | top 6 | measured | measured | coverage ceiling | measured |

No numerical values should be hard-coded into the notebook specification.

The exact committed notebook must record whatever the verified runtime actually observes.

In particular, the release gate MUST NOT require that reranking improve the first-stage ranking. A measured degradation is a valid workshop result and should be explained rather than hidden.

---

# 19. Error analysis

The notebook MUST automatically select illustrative examples from four categories where available:

1. **Retriever already correct; reranker preserves it**
2. **Retriever top-1 wrong; reranker fixes it**
3. **Retriever top-1 correct; reranker makes it worse**
4. **Gold document missing from top-k; reranker cannot recover it**

For each example show:

```text
query
gold intent
retriever top-k
retriever scores
reranked top-k
reranker scores
result category
```

This is more educational than reporting aggregate metrics alone.

The notebook SHOULD explicitly invite the learner to inspect why semantically similar banking intents can be confused.

---

# 20. Workshop exercise

The notebook MUST include at least one non-blocking learner activity.

Recommended exercise:

> Before revealing the reranked order for one selected query, inspect the embedding shortlist and predict:
>
> - whether the gold document is present;
> - whether the reranker is likely to change the top result; and
> - what type of ambiguity the first-stage retriever encountered.

The answer is then revealed by an already complete executable cell.

No `TODO`, incomplete code cell, or `WRITE ME` placeholder is permitted.

---

# 21. Optional rerank-depth experiment

Disabled by default:

`RUN_K_SWEEP = False`

When enabled, evaluate:

```text
K = 3
K = 6
K = 10
```

Report for each:

- shortlist coverage;
- final recall@1;
- final MRR;
- number of cross-encoder pair evaluations;
- measured reranker wall time.

The key interpretation is:

```text
larger K
→ potentially higher candidate recall
→ more expensive reranking
```

The experiment MUST be presented as a tutorial quality/compute tradeoff, not an optimization result.

The default `K=6` must remain unchanged by the experiment.

---

# 22. BYOD contract

BYOD is optional and disabled on the default path.

## Documents file

CSV with:

```text
doc_id,text
```

Requirements:

- 2–1,000 documents;
- unique non-empty `doc_id`;
- non-empty text;
- each document ≤ 100,000 characters;
- each row represents one retrievable unit.

The notebook does **not** perform automatic long-document chunking.

Users with long reports must pre-chunk them into meaningful retrievable units.

---

## Queries file

CSV with:

```text
query_id,query
```

Optional evaluation field:

```text
gold_doc_id
```

If `gold_doc_id` is supplied:

- every value must reference an existing `doc_id`;
- ranking metrics are calculated.

If labels are absent:

```text
evaluation verdict = not-measurable
```

The notebook still performs retrieval/reranking and exports results.

Recommended workshop guard:

```text
≤ 500 BYOD queries
≤ 1,000 documents
```

A stricter tutorial ceiling MAY be used than the underlying model API to prevent accidental multi-hour reranking jobs, provided the difference is explicitly documented.

---

# 23. BYOD privacy warning

Before the BYOD section, state that the standalone workflow processes data locally inside the selected notebook runtime and does not intentionally send corpus/query content to DIMER services.

Users MUST be warned not to upload:

- confidential material;
- restricted data;
- personal information;
- regulated records; or
- data they are not authorized to process

to a hosted Colab/Kaggle environment.

Downloading model weights from Hugging Face does not require transmitting the user's search corpus to Hugging Face.

---

# 24. Machine-readable outputs

Write under:

`outputs/qwen3_semantic_search/`

Required outputs:

### `document_embeddings.npz`

Contains:

```text
doc_ids
embeddings
```

Vectors MUST remain mapped to their document identifiers.

### `retrieval_results.csv`

One row per retained retrieval result:

```text
query_id
doc_id
retrieval_rank
retrieval_score
is_gold
```

### `reranked_results.csv`

```text
query_id
doc_id
retrieval_rank
retrieval_score
reranked_rank
reranker_score
is_gold
```

### `metrics.json`

Contains:

```text
random baseline
lexical baseline
retriever metrics
shortlist coverage
pipeline metrics
conditional reranker metrics
help/hurt/unchanged/unrecoverable counts
timings
```

### `provenance.json`

Contains:

```text
notebook specification version
notebook profile
pedagogical mode
embedding model ID
embedding revision
embedding weight digest
reranker model ID
reranker revision
reranker weight digest
dataset source
dataset source revision
dataset file digests
sample seed
sample digest
RERANK_K
instructions
Python version
PyTorch version
Transformers version
device
dtype
runtime timings
```

No secrets or tokens may appear in provenance.

---

# 25. Terminal summary

The final executable cell MUST print a concise terminal summary similar in structure to:

```text
DIMER Qwen3 Semantic Search + Reranking Workshop
------------------------------------------------
Queries evaluated: ...
Documents indexed: ...
Rerank depth: ...

Lexical recall@1: ...
Embedding recall@1: ...
Embedding recall@6 / shortlist coverage: ...
Two-stage recall@1: ...
Two-stage MRR: ...

Reranking effects:
  helped: ...
  hurt: ...
  unchanged: ...
  unrecoverable: ...

Outputs: outputs/qwen3_semantic_search/
```

The values must come from the current execution.

---

# 26. Interpretation and limitations

The final interpretation section MUST cover the following.

## Retrieval and reranking are different operations

Embedding retrieval efficiently identifies candidates. Cross-encoder reranking spends more computation evaluating interactions between each query and candidate.

## Reranking does not solve candidate recall

A relevant document omitted from the first-stage shortlist is unavailable to the reranker.

## Similarity and relevance scores are not probabilities

Cosine similarity and Qwen3 reranker scores are ordering signals. Neither is calibrated as a probability of relevance.

## Banking77 is tutorial evidence

The experiment uses short banking-intent phrases as candidate documents and a single gold intent per query.

This does not establish performance on:

- long documents;
- enterprise search;
- multilingual corpora;
- graded relevance;
- large-scale retrieval;
- production query distributions.

## Pretraining overlap is unknown

Banking77 or related content may have appeared in upstream pretraining. The notebook cannot establish absence of contamination.

## Real search may have several relevant documents

The sample has one gold label. Real information retrieval commonly requires graded or multiple relevance judgments.

## Scaling behavior differs by stage

Document embeddings can be precomputed.

Cross-encoder compute scales with approximately:

`number of queries × shortlist depth`

This is why the reranker is normally applied after candidate retrieval rather than across the whole corpus.

---

# 27. Explicit non-goals

The notebook MUST NOT include:

- Qwen fine-tuning;
- adapter generation;
- model selection using the test queries;
- a vector database;
- approximate nearest-neighbor indexing;
- automatic document chunking;
- RAG answer generation;
- Qwen3-0.6B or SmolLM2 generation;
- production DIMER API calls;
- arbitrary relevance thresholds;
- claims of calibrated confidence;
- benchmark claims from the Banking77 tutorial results.

Fine-tuning remains covered by:

- `qwen3_embedding_colab.ipynb`;
- `qwen3_reranker_colab.ipynb`.

A later **RAG / grounded generation workshop** can build on the output of this notebook as a separate capability.

---

# 28. Notebook cell plan

| # | Type | Section | Default execution |
|---:|---|---|---|
| 0 | Markdown | Title, profile, workshop contract, learning objectives | Yes |
| 1 | Markdown | How two-stage retrieval works | Yes |
| 2 | Code | Form parameters | Yes |
| 3 | Markdown | Runtime and dependency contract | Yes |
| 4 | Code | Install/verify pinned environment | Yes |
| 5 | Markdown | Model provenance and supply-chain boundary | Yes |
| 6 | Code | Embedded manifests + staging/digest helpers | Yes |
| 7 | Markdown | Dataset provenance and limitations | Yes |
| 8 | Code | Fetch and digest-check Banking77 | Yes |
| 9 | Markdown | Validation contract | Yes |
| 10 | Code | Build 77-doc corpus + 154-query evaluation set; validate | Yes |
| 11 | Markdown | Baselines | Yes |
| 12 | Code | Random floor + lexical retrieval | Yes |
| 13 | Markdown | Stage 1: bi-encoder retrieval | Yes |
| 14 | Code | Load Qwen3 Embedding and build document index | Yes |
| 15 | Code | Embed queries and retrieve all 77 documents | Yes |
| 16 | Markdown | Interpret retriever metrics | Yes |
| 17 | Code | Compute retrieval metrics + shortlist coverage | Yes |
| 18 | Markdown | Workshop prediction exercise | Yes |
| 19 | Markdown | Stage 2: cross-encoder reranking | Yes |
| 20 | Code | Load Qwen3 Reranker and rerank top-6 | Yes |
| 21 | Markdown | End-to-end evaluation semantics | Yes |
| 22 | Code | Pipeline + conditional reranker metrics | Yes |
| 23 | Markdown | Where did reranking help? | Yes |
| 24 | Code | Automatic help/hurt/unrecoverable case analysis | Yes |
| 25 | Markdown | Quality/compute tradeoff | Yes |
| 26 | Code | Optional `K` sweep | Default no-op |
| 27 | Markdown | Export contract | Yes |
| 28 | Code | Write NPZ/CSV/JSON/provenance | Yes |
| 29 | Markdown | Bring Your Own Data | Yes |
| 30 | Code | Optional BYOD branch | Default no-op |
| 31 | Markdown | Limitations and transfer questions | Yes |
| 32 | Code | Terminal summary + output existence assertions | Yes |

Every executable cell MUST be preceded by explanatory markdown.

The committed notebook MUST contain no persisted execution counts or outputs.

---

# 29. Required assertions

The default path SHOULD fail early on structural defects and include at least the following assertions:

```text
dataset file digests valid
77 unique intent documents
154 balanced held-out queries
all gold_doc_ids exist
embedding model identity matches manifest
reranker model identity matches manifest
embedding vectors shape = (77, 1024)
embedding vectors finite
embedding norms approximately 1
retrieval rankings contain unique document IDs
each shortlist contains exactly RERANK_K candidates
reranker scores finite and within [0, 1]
every final rank maps to an original candidate
shortlist_coverage >= pipeline_recall@1
all required output files exist
```

The notebook MUST NOT assert that reranking improves recall@1 or MRR.

Whether it helps is an empirical result of the execution.

---

# 30. Release verification

The exact committed notebook blob must execute top-to-bottom in a clean supported runtime.

Preferred qualification environment:

**Kaggle Tesla T4**, matching the existing Qwen3 carrier qualification class.

The release run must begin with:

- no repository checkout;
- empty notebook working directory apart from the notebook;
- empty model snapshot directories;
- empty Banking77 cache;
- no authentication token.

Record:

```text
notebook commit
notebook blob SHA
execution date
platform
Python
PyTorch
Transformers
CUDA/device
both model IDs/revisions
model files downloaded and verified
dataset files downloaded and verified
number of corpus documents
number of evaluation queries
RERANK_K
all observed metrics
stage-level timings
outputs produced
cell success count
wall time
```

Release readiness requires:

- every cell succeeding;
- both immutable snapshots verified;
- all default data validation succeeding;
- complete lexical → embedding → reranking flow;
- all required exports;
- no repository imports;
- no external DIMER service calls;
- no authentication prompt;
- no user interaction; and
- no unsupported claims in the interpretation text.

Observed ranking metrics are evidence that the notebook executed correctly on the stated sample. They are not production or benchmark evidence.

---

# 31. Recommended repository placement

This is a **cross-model workshop**, not the primary tutorial of either model carrier.

It SHOULD have one canonical copy rather than being duplicated between the embedding and reranker repositories.

Recommended location:

```text
ml-worker/
  integrations/
    dimer/
      workshops/
        qwen3-semantic-search/
          DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb
          README.md
          docs/
            release-verification.md
          tools/
            build_notebook.py
            validate_notebook.py
```

The two model repositories may then link to the canonical workshop.

This avoids weakening the current carrier repositories' one-primary-tutorial release checks and prevents two copies of the workshop from drifting apart.

---

# 32. Follow-on workshop

The logical next notebook after this one is:

**Building a Grounded Retrieval-Augmented Application with DIMER**

That notebook could consume the ranked documents produced here and add a generation model such as Qwen3-0.6B or SmolLM2-360M-Instruct.

Generation SHOULD remain separate from this workshop so learners can first understand and measure the retrieval system independently of answer-generation quality.

---

# 33. Workshop learning arc

The intended conceptual sequence is:

**Lexical search**  
↓  
*Words overlap, but meaning is only weakly represented.*

**Dense semantic retrieval**  
↓  
*The system searches the entire corpus efficiently using reusable vectors.*

**Candidate shortlist**  
↓  
*Retrieval recall establishes the ceiling for downstream ranking.*

**Cross-encoder reranking**  
↓  
*More compute is spent only on the few candidates worth examining closely.*

**End-to-end evaluation**  
↓  
*Search quality is a property of the composed system, not either model in isolation.*

That system-level lesson is the core reason this notebook exists.
## Guided and optional-path follow-up

Adopt the non-breaking fleet 2.2 guided layer while retaining the 2.1 profile declaration. Optional learning uses the existing isolated depth sweep, with no test-based canonical retuning. BYOD requires either all valid labels or no labels, unloads one model before loading the other, and writes distinct rankings/results plus input/model/runtime provenance. A manual-restart bootstrap is not evidence of the uninterrupted no-interaction Run-all requirement; that qualification gap remains open.
