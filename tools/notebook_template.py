"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, samples.py, metrics.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E contrastive-adaptation workflow: the pinned Qwen3-Embedding-0.6B snapshot is
digest-verified and loaded, a digest-pinned real corpus (Banking77 messages paired with their intent phrases)
is fetched, validated and split, the embedding contract is exercised on queries and documents, the frozen
model's retrieval metrics on held-out queries are read beside a random floor and a lexical baseline, a bounded
contrastive fine-tuning of the last decoder layers adapts the embedder in the kernel, the held-out split is
scored again, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "qwen3_embedding_pipeline",
    "repo_name": "qwen3-embedding-pipeline",
    "stem": "qwen3_embedding",
    "notebook_name": "qwen3_embedding_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned Qwen3-Embedding-0.6B snapshot (safetensors, 1.19 GB), fetches the two digest-pinned Banking77 CSV files "
        "from the project repository (1.1 MB, no credential), pairs every customer message with its intent phrase and draws "
        "616 / 154 / 385 training, validation and test pairs balanced over the 77 intents from the release's own partition, "
        "embeds three test queries and the 77 intent documents through the inference contract with an input manifest and a "
        "rejection probe, scores the frozen embedder on the test queries by recall@1 / recall@5 / recall@10 and MRR over "
        "the 77 documents beside the random floor and a lexical (token-overlap) baseline, runs a bounded contrastive "
        "fine-tuning (InfoNCE with in-batch negatives) of the last two decoder layers with validation-MRR epoch "
        "selection, scores the held-out split again, retrieves for the same three queries with the adapted embedder, "
        "exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify "
        "parity. The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog "
        "and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about four minutes of model time "
        "after the downloads; a CUDA runtime is used automatically when present (bfloat16 there, float32 on CPU)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "query–positive pairs as a CSV (columns `id`, `query`, `positive`, optional `negative`), a JSON array or a JSONL file "
        "of `{{id, query, positive}}` records — the document set is the unique positives (and negatives); set `INSTRUCTION` "
        "to a one-line description of your retrieval task. They pass through the same validation, seeded query-disjoint "
        "split, floor and baseline, frozen scoring, fine-tuning, held-out evaluation, retrieval, artifact export and "
        "reload-parity cells as the Banking77 sample. The expected schema and the ceilings are stated in the Prerequisites "
        "and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "Qwen3EmbeddingPipeline",
    "weights_key": "qwen3-embedding-0.6b",
    "modules": ["pipeline.py", "samples.py", "metrics.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "Qwen3-Embedding-0.6B — DIMER E2E contrastive fine-tuning tutorial: intent retrieval on Banking77 (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/qwen3-embedding-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/qwen3-embedding-pipeline/blob/main/tutorials/qwen3_embedding_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--Embedding--0.6B-ffcc4d?style=flat",
            "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-QwenLM%2FQwen3--Embedding-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/QwenLM/Qwen3-Embedding",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2506.05176-b31b1b.svg", "https://arxiv.org/abs/2506.05176"),
    ],
    "capability": "text embeddings (1024-d, last-token pooled, L2-normalised, instruction-aware queries) and bounded contrastive fine-tuning of the last decoder layers on query–positive pairs, measured by held-out retrieval recall@k and MRR, using the pinned `Qwen/Qwen3-Embedding-0.6B` weights",
    "intro": (
        "At inference each text is tokenised with left padding and truncated at 8,192 tokens, a 0.6 B-parameter Qwen3 "
        "decoder encodes it, the hidden state of the **last token** is taken as the text's vector, and the pipeline "
        "L2-normalises it to unit length. Queries are prefixed with a task instruction (`Instruct: …\\nQuery:`) because "
        "the model is instruction-aware; documents are embedded as-is. **Embeddings are representations, not "
        "predictions:** a vector carries no score, and the carried pipeline module adds snapshot verification, input "
        "validation with named ceilings, the query/document formatting contract, a fixed output contract and the "
        "`cosine_similarity`, `validate_inputs` and `evaluation_report` helpers.\n\n"
        "What this notebook adds to inference is **adaptation measured by retrieval**. The dataset is real: Banking77 "
        "(Casanueva et al., 2020; CC BY 4.0) ships 13,083 customer-support messages labelled with 77 fine-grained banking "
        "intents as two digest-pinned CSV files fetched from the project repository at a pinned commit. Every intent name "
        "becomes a short **document** (`card_arrival` → `card arrival`) and every message a **query** whose positive is "
        "its intent phrase, so nearest-neighbour retrieval over the 77 documents is intent detection — a task the embedder "
        "was never tuned for, with documents that are two or three words long. The carried `metrics.py` ranks the "
        "documents for every held-out query by cosine and reads the rank of its positive: **recall@1**, **recall@5**, "
        "**recall@10** and **MRR**; a **random floor** (1 / 77 recall@1) and a **lexical baseline** (Jaccard token "
        "overlap between message and phrase) frame the frozen number. The fine-tuning question is whether a bounded "
        "contrastive adaptation of the last decoder layers on 616 pairs raises retrieval on messages the model has not "
        "seen. Nothing here is a quality claim about your retrieval task: it is one seeded split of one corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real corpus and validate and split it "
        "without leakage; embed queries and documents through the public API with the instruction contract and read "
        "the vector contract correctly; read recall@k and MRR beside a random floor and a lexical baseline and "
        "understand what they do and do not measure; run a bounded contrastive fine-tuning with explicit "
        "hyperparameters and validation-based epoch selection; evaluate on an independent test split; compare retrieved "
        "documents before and after; and export a safetensors adapter that reloads against the pinned base with "
        "verified parity."
    ),
    "exclusions": (
        "reranking (see the sibling Qwen3 reranker pipeline), text generation or chat, classification heads, clustering "
        "quality, Matryoshka dimension truncation (fixed at 1024 here), hard-negative mining beyond the in-batch and "
        "explicit negatives of the pair contract, full-model or embedding-table training, and any claim that a Banking77 "
        "intent split stands in for your retrieval task. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available. **Precision differs by device:** the pipeline runs float32 on CPU and bfloat16 on CUDA, so cosine values and the recorded metrics can differ between the two. CPU is adequate for this sample: the build record measured about 5 s to load and digest-verify the 1.19 GB snapshot, about 30 s to embed the 385 test queries and 77 documents, and about 47 s per training epoch over 616 pairs plus a validation pass per epoch. The pinned `torch==2.14.0` install and the 1.19 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and NumPy; what a dense vector, a unit norm and cosine similarity are; what recall@k and mean reciprocal rank measure and why they need a labelled query–document set; what a contrastive (InfoNCE) loss with in-batch negatives does.",
        "- **Data contract:** records are `{{id, query, positive}}` — a query of 1..100,000 characters (text beyond 8,192 tokens is truncated and flagged at inference; queries and documents are truncated to 64 tokens **during training only**), a positive document of 1..1,000 characters, an optional `negative` document (added to the document set), ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a dataset needs 8..20,000 records and at least 2 distinct positives; queries are de-duplicated case-insensitively before splitting so the same message never sits in two splits. BYOD accepts CSV, JSON or JSONL in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a positive is relevant to its query or that the instruction describes the task — a mislabelled pair set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal query log with its relevance labels is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches two pinned objects (`train.csv` 839,073 bytes, `test.csv` 239,961 bytes; SHA-256 `b06e26ac…` / `d12d6e3b…`) from `raw.githubusercontent.com` at the pinned `PolyAI-LDN/task-specific-datasets` commit over HTTPS, each refused on any mismatch before it is read; Banking77 is CC BY 4.0 (Casanueva et al., 2020).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and split\n\n"
                "`fetch_corpus` downloads the two pinned Banking77 CSV files (or reads them from the cache), refuses a "
                "byte-size or SHA-256 mismatch per file before it is parsed, and `read_corpus` checks the columns, row "
                "counts and the 77 intents of each member. `build_sample_dataset` turns every message into a "
                "`{{id, query, positive}}` pair whose positive is the intent name as words, drops repeated messages, and draws "
                "8 training and 2 validation pairs per intent from the `train` member (disjoint messages) and 5 test pairs "
                "per intent from the `test` member by a seeded shuffle — the release's own partition, balanced over all 77 "
                "intents. `validate_dataset` then checks every record against the contract, `documents` lists the 77 "
                "retrieval candidates, `check_split_disjoint` asserts no message appears in two splits, and the training "
                "split is written to `outputs/{stem}_train.csv` in the shape BYOD expects. `INSTRUCTION` is the task "
                "description every query carries in later cells.\n\n"
                "Look for: 10,003 + 3,080 raw rows, two digests, splits 616 / 154 / 385, 77 documents, and four refusal "
                "probes — a duplicate id, an empty query, a missing field and a dataset too small to split — each rejected "
                "before `torch` does anything."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n"
                "INSTRUCTION = 'Given a customer support message, retrieve the banking intent it expresses'  # @param {{type:\"string\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/banking77'))\n"
                "    raw_rows = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "document_set = documents([*train_records, *val_records, *test_records])\n"
                "disjoint = check_split_disjoint(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'instruction': INSTRUCTION, 'raw_rows': raw_rows, 'splits': disjoint, 'n_documents': len(document_set), 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_queries': manifest['unique_queries'], 'n_documents': manifest['n_documents'], 'query_chars': manifest['query_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'example': train_records[0], 'documents': document_set[:6] + ['...']}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty query': [{{**train_records[0], 'query': '   '}}, *train_records[1:8]],\n"
                "    'missing field': [{{'id': r['id'], 'query': r['query']}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Embed through the inference contract\n\n"
                "Before any adaptation, the embedding contract is exercised as it always was. `validate_inputs` applies "
                "exactly the checks `embed` applies — both route through the same private `_check_inputs` — so type, "
                "batch size 1..`MAX_BATCH`, non-empty text, the character ceiling, a `kind` in `KINDS` and a non-empty "
                "instruction are enforced identically; it returns an input manifest for the 77 documents with the three "
                "probe queries recorded under `query_manifest`, and a deliberately oversized batch is validated too and its "
                "rejection recorded as a finding. `embed` returns one unit-norm 1024-d vector per text with `n_tokens` and "
                "`truncated` flags; queries carry the instruction prefix, documents do not. The three test queries are "
                "ranked against the 77 documents by cosine and their top-3 documents printed beside the gold intent — a "
                "qualitative look before any metric is read; **cosine is a similarity, not a probability**, and no "
                "threshold ships. The frozen top-3 lists are kept as the *before* column for Section 9."
            ),
            "code": (
                "import time\n\n"
                "probe_records = test_records[:3]\n"
                "probe_queries = [r['query'] for r in probe_records]\n"
                "print({{'ceilings': {{'MAX_BATCH': MAX_BATCH, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'EMBEDDING_DIM': EMBEDDING_DIM, 'MAX_TRAIN_TOKENS': MAX_TRAIN_TOKENS, 'KINDS': KINDS}}}})\n"
                "input_manifest = validate_inputs(document_set[:MAX_BATCH], 'document', names=[f'doc-{{i:02d}}' for i in range(min(len(document_set), MAX_BATCH))])\n"
                "input_manifest['query_manifest'] = validate_inputs(probe_queries, 'query', INSTRUCTION, names=[r['id'] for r in probe_records])\n"
                "try:\n"
                "    validate_inputs(['probe'] * (MAX_BATCH + 1))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-batch-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n\n"
                "def top_documents(vectors, doc_vectors, k=3):\n"
                "    scores = np.asarray(vectors, dtype=np.float32) @ np.asarray(doc_vectors, dtype=np.float32).T\n"
                "    return [[(document_set[j], round(float(row[j]), 4)) for j in np.argsort(-row, kind='stable')[:k]] for row in scores]\n\n"
                "started = time.perf_counter()\n"
                "doc_rows = []\n"
                "for start in range(0, len(document_set), MAX_BATCH):\n"
                "    doc_rows.extend(pipe.embed(document_set[start:start + MAX_BATCH], kind='document')['embeddings'])\n"
                "document_seconds = round(time.perf_counter() - started, 3)\n"
                "started = time.perf_counter()\n"
                "query_result = pipe.embed(probe_queries, kind='query', instruction=INSTRUCTION)\n"
                "query_seconds = round(time.perf_counter() - started, 3)\n"
                "doc_vectors = np.asarray(doc_rows, dtype=np.float32)\n"
                "query_vectors = np.asarray(query_result['embeddings'], dtype=np.float32)\n"
                "checks = {{\n"
                "    'one_vector_per_text': doc_vectors.shape == (len(document_set), EMBEDDING_DIM) and query_vectors.shape == (3, EMBEDDING_DIM),\n"
                "    'unit_norm': bool(np.allclose(np.linalg.norm(doc_vectors, axis=1), 1.0, atol=1e-4)) and bool(np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0, atol=1e-4)),\n"
                "    'contract_fields': query_result['dim'] == EMBEDDING_DIM and query_result['pooling'] == POOLING and query_result['normalized'] is True and query_result['kind'] == 'query' and query_result['instruction'] == INSTRUCTION,\n"
                "    'nothing_truncated': not any(query_result['truncated']),\n"
                "    'all_values_finite': bool(np.isfinite(doc_vectors).all()) and bool(np.isfinite(query_vectors).all()),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'embed output failed a sanity check: {{checks}}')\n"
                "before = dict(zip([r['id'] for r in probe_records], top_documents(query_vectors, doc_vectors), strict=True))\n"
                "for record in probe_records:\n"
                "    print({{'id': record['id'], 'query': record['query'][:80], 'gold': record['positive'], 'frozen_top3': before[record['id']]}})\n"
                "print({{'n_tokens': query_result['n_tokens'], 'seconds': {{'documents': document_seconds, 'queries': query_seconds}}, 'checks': checks, 'findings': len(input_manifest['findings']), 'score_semantics': 'cosine between unit vectors; a similarity, not a probability; no threshold shipped'}})"
            ),
        },
        {
            "md": (
                "## 6. The random floor, the lexical baseline and the frozen model on the test split\n\n"
                "Three numbers frame the adaptation, all over the same 77-document set. The **random floor** is what a "
                "uniformly random ranking achieves in expectation (recall@1 = 1 / 77, MRR ≈ 0.064). The **lexical "
                "baseline** ranks the documents for each query by Jaccard overlap of lower-cased tokens — what a system "
                "with no model gets from shared words such as *card* or *pin*. `pipe.evaluate` embeds the 77 documents "
                "once and every test query with `INSTRUCTION`, ranks by cosine, and reads the rank of each query's own "
                "positive; ties are counted against the positive. Look for the frozen embedder well above both — the "
                "build record saw recall@1 in the sixties and MRR in the seventies — and read `median_rank` beside the "
                "means. About half a minute on CPU."
            ),
            "code": (
                "def brief(m):\n"
                "    return {{k: round(m[k], 4) for k in ('recall@1', 'recall@5', 'recall@10', 'mrr')}} | {{'median_rank': m.get('median_rank')}}\n\n"
                "floor = random_floor(len(document_set))\n"
                "print({{'random_floor': {{k: round(floor[k], 4) for k in ('recall@1', 'recall@5', 'recall@10', 'mrr')}}, 'baseline': floor['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "baseline_lexical = pipe.lexical_baseline(test_records, document_set)\n"
                "print({{'lexical_baseline': brief(baseline_lexical), 'baseline': baseline_lexical['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "print({{'frozen_model_test': brief(frozen_test), 'n_queries': frozen_test['n_queries'], 'n_documents': frozen_test['n_documents'], 'verdict': frozen_test['verdict'], 'adapted': frozen_test['adapted'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "assert frozen_test['n_documents'] == baseline_lexical['n_documents'] and frozen_test['mrr'] > floor['mrr']"
            ),
        },
        {
            "md": (
                "## 7. Bounded contrastive fine-tuning\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_LAYERS` decoder layers — two by default, "
                "31,461,888 of 595,776,512 parameters; the token embeddings, the earlier layers and the final norm stay "
                "frozen — with an **InfoNCE** loss: each batch embeds its queries (with `INSTRUCTION`) and the unique "
                "documents among its positives in one forward pass, and every query must pick its own positive out of the "
                "batch's documents by cosine at `TEMPERATURE` — the other queries' positives are the negatives. AdamW at a "
                "fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler; texts are truncated to "
                "`MAX_TRAIN_TOKENS` (64) **during training only**. Epoch 0 records the frozen model's validation retrieval "
                "metrics over the validation document set; every epoch is scored the same way, and the epoch with the "
                "highest validation MRR is kept.\n\n"
                "Watch validation recall@1 climb by ten points or so over two epochs while recall@5 passes 95 % (about "
                "47 s of training plus a validation pass per epoch on CPU). The build record's sweep on this sample: two layers at 2e-5 reached recall@1 77.1 %, four layers at 2e-5 82.1 % with a 252 MB adapter, two layers at 5e-5 80.3 % with a 126 MB adapter — the default."
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 2  # @param {{type:\"integer\"}}\n"
                "TEMPERATURE = 0.05  # @param {{type:\"number\"}}\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_recall@1'] = round(entry['val']['recall@1'], 4)\n"
                "        row['val_recall@5'] = round(entry['val']['recall@5'], 4)\n"
                "        row['val_mrr'] = round(entry['val']['mrr'], 4)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, instruction=INSTRUCTION, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_layers=TRAINABLE_LAYERS, temperature=TEMPERATURE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'objective': adapt_result['objective'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'train_documents': adapt_result['n_train_documents'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no message in it appears in the "
                "training or validation splits. The adapted embedder is scored exactly as the frozen one was in Section 6 "
                "— same queries, same 77 documents, same instruction — and the four rows are put side by side. Look for "
                "recall@1 up by ten points or more and MRR up by about a tenth; the cell asserts the adapted MRR is above "
                "the frozen MRR. 385 queries from one seeded split of one corpus give no dispersion estimate; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain on Banking77 "
                "intents says nothing about your retrieval task until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "adapted_val = pipe.evaluate(val_records, instruction=INSTRUCTION, candidates=documents(val_records))\n"
                "comparison = {{\n"
                "    metric: {{'random_floor': round(floor[metric], 4), 'lexical': round(baseline_lexical[metric], 4), 'frozen': round(frozen_test[metric], 4), 'adapted': round(adapted_test[metric], 4)}}\n"
                "    for metric in ('recall@1', 'recall@5', 'recall@10', 'mrr')\n"
                "}}\n"
                "comparison['median_rank'] = {{'lexical': baseline_lexical['median_rank'], 'frozen': frozen_test['median_rank'], 'adapted': adapted_test['median_rank']}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('recall@1', 'recall@5', 'recall@10', 'mrr')}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'instruction': INSTRUCTION,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'n_documents': len(document_set),\n"
                "    'baselines': {{'random_floor': floor, 'lexical': baseline_lexical}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['mrr'] > frozen_test['mrr']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Retrieve before and after, export the adapter and reload it\n\n"
                "The three test queries embedded by the frozen model in Section 5 are embedded again by the adapted model "
                "through the same `embed` contract, ranked against the re-embedded 77 documents, and their top-3 lists "
                "printed side by side with the gold intent. Read them as observations: the metric is Section 8, and the "
                "adapter moves the last decoder layers so **every vector changes** — each document's cosine to its frozen "
                "self is printed too. The per-batch `evaluation_report` helper — the inference-stage helper — is written "
                "for the probe queries and stays `not-measurable`, because a batch of vectors has no metric without a "
                "labelled set; `pipe.evaluate` is that labelled evaluation.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two decoder layers, about 126 MB — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the instruction it was trained with, the tensor "
                "names, the file size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`Qwen3EmbeddingPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest and "
                "digest **before** deserialising, refuses any tensor that is not a decoder-layer tensor of the base, and "
                "overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). "
                "The cell asserts identical query vectors and an identical test MRR (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "doc_rows_after = []\n"
                "for start in range(0, len(document_set), MAX_BATCH):\n"
                "    doc_rows_after.extend(pipe.embed(document_set[start:start + MAX_BATCH], kind='document')['embeddings'])\n"
                "doc_vectors_after = np.asarray(doc_rows_after, dtype=np.float32)\n"
                "query_vectors_after = np.asarray(pipe.embed(probe_queries, kind='query', instruction=INSTRUCTION)['embeddings'], dtype=np.float32)\n"
                "after = dict(zip([r['id'] for r in probe_records], top_documents(query_vectors_after, doc_vectors_after), strict=True))\n"
                "rows = []\n"
                "for record in probe_records:\n"
                "    rows.append({{'id': record['id'], 'query': record['query'], 'gold': record['positive'], 'frozen_top3': ' | '.join(d for d, _s in before[record['id']]), 'adapted_top3': ' | '.join(d for d, _s in after[record['id']]), 'gold_rank_frozen': next((i + 1 for i, (d, _s) in enumerate(before[record['id']]) if d == record['positive']), None), 'gold_rank_adapted': next((i + 1 for i, (d, _s) in enumerate(after[record['id']]) if d == record['positive']), None)}})\n"
                "    print({{k: rows[-1][k] for k in ('id', 'gold', 'frozen_top3', 'adapted_top3', 'gold_rank_frozen', 'gold_rank_adapted')}})\n"
                "document_shift = {{'self_cosine_min': round(float(np.min(np.sum(doc_vectors * doc_vectors_after, axis=1))), 4), 'self_cosine_median': round(float(np.median(np.sum(doc_vectors * doc_vectors_after, axis=1))), 4)}}\n"
                "single_report = evaluation_report(pipe.embed(probe_queries, kind='query', instruction=INSTRUCTION), sample_kind='three Banking77 test queries' if not USE_BYOD else 'three BYOD test queries')\n"
                "print({{'document_shift': document_shift, 'batch_report_verdict': single_report['verdict'], 'probes_with_changed_top3': sum(r['frozen_top3'] != r['adapted_top3'] for r in rows), 'of': len(rows)}})\n"
                "with open('outputs/{stem}_retrieval.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'instruction': artifact_manifest['adapter']['instruction'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = Qwen3EmbeddingPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_queries = np.asarray(reloaded.embed(probe_queries, kind='query', instruction=INSTRUCTION)['embeddings'], dtype=np.float32)\n"
                "reloaded_test = reloaded.evaluate(test_records[:77], instruction=INSTRUCTION, candidates=document_set)\n"
                "in_memory_test = pipe.evaluate(test_records[:77], instruction=INSTRUCTION, candidates=document_set)\n"
                "parity = {{'query_vectors_identical': bool(np.array_equal(query_vectors_after, reloaded_queries)), 'mrr_in_memory': round(in_memory_test['mrr'], 6), 'mrr_reloaded': round(reloaded_test['mrr'], 6)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['query_vectors_identical'] and abs(in_memory_test['mrr'] - reloaded_test['mrr']) < 1e-9\n\n"
                "weight_entry = next(entry for entry in snapshot['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'instruction': INSTRUCTION,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {{k: {{'name': v[0], 'bytes': v[1], 'sha256': v[2]}} for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_queries': probe_queries, 'n_tokens': query_result['n_tokens'], 'seconds': {{'documents': document_seconds, 'queries': query_seconds}}}},\n"
                "    'comparison': comparison,\n"
                "    'retrieval_before_after': rows,\n"
                "    'document_shift': document_shift,\n"
                "    'batch_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'bfloat16' if pipe.device.startswith('cuda') else 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen embedder already retrieves the right intent phrase for well over half of unseen messages (recall@1 in "
        "the sixties against a lexical baseline in the thirties and a random floor of 1.3 %), and a bounded contrastive "
        "fine-tuning of the last two decoder layers on 616 pairs lifts held-out recall@1 by more than ten points and "
        "MRR by about a tenth in a few minutes on CPU, with a 126 MB adapter that reloads to identical vectors. That is "
        "the claim: the adaptation contract can adapt the embedder to a retrieval task end to end on a real labelled "
        "corpus, and the numbers it produces are read against a random floor, a lexical baseline and the frozen model "
        "rather than in isolation.\n\n"
        "The test split is 385 queries against 77 two-word documents from one seeded split of one corpus with no "
        "dispersion estimate; recall@k and MRR say whether the gold document ranks first, not whether the vectors are good "
        "for any other use. The adapter changes the last layers, which every input shares, so every vector shifts (Section "
        "9 prints each document's cosine to its frozen self) and cosine values from the adapted model are not comparable "
        "to the frozen model's; nothing here measures the effect on other tasks. On CUDA the model runs in bfloat16 and the "
        "recorded float32 CPU numbers will not reproduce to the last digit.\n\n"
        "Three things to carry to real data. **Floors first:** the random floor and the lexical baseline on *your* "
        "documents are the numbers to read before any embedder's; a lexical baseline near the frozen model means the task "
        "is mostly keyword matching. **Leakage:** de-duplicate queries across splits (the contract does this "
        "case-insensitively) and split by user or session when your queries come from one. **Documents:** the document set "
        "here is the label vocabulary; a real retrieval task has many more documents than intents and needs hard "
        "negatives beyond the in-batch ones this contract uses.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled corpus, "
        "validate the demonstrated dataset contract without leakage, execute the embedding contract and a bounded "
        "contrastive fine-tuning, evaluate by retrieval metrics against a random floor, a lexical baseline and the frozen "
        "model on an independent split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, embedding quality on any other task, a usable "
        "acceptance threshold, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_LAYERS = 1` and watch the gain "
        "shrink; set `TEMPERATURE = 0.2` and read whether the softer contrast learns as fast; set `EPOCHS = 4` and watch "
        "whether validation MRR keeps rising or turns (the best epoch is kept either way); change `INSTRUCTION` and re-read "
        "the frozen numbers — the model is instruction-aware; or bring your own pairs through BYOD and read the lexical "
        "baseline before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/QwenLM/Qwen3-Embedding\n"
        "- Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models (2025): https://arxiv.org/abs/2506.05176\n"
        "- Efficient Intent Detection with Dual Sentence Encoders (Casanueva et al., 2020; Banking77, CC BY 4.0): https://arxiv.org/abs/2003.04807\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
