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
    # SWP-R (2026-10-05 fleet sweep): nothing is pip-installed into the notebook kernel. The fleet's uv isolated-environment
    # mechanism (build_notebook.py/2.2): managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the
    # pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28
    # --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt` (uv 0.12.15).
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
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
        "and no configuration edit (NOTEBOOK_SPEC 2.2 §5). On CPU the whole path takes about four minutes of model time "
        "after the downloads; a CUDA runtime is used automatically when present (bfloat16 there, float32 on CPU)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "query–positive pairs — as `BYOD_PATH` (a path in the runtime, which works on Colab, Kaggle and Jupyter) or, when it is empty, "
        "through the Colab upload dialog — as a CSV (columns `id`, `query`, `positive`, optional `negative`), a JSON array or a JSONL file "
        "of `{id, query, positive}` records — the document set is the unique positives (and negatives); set `INSTRUCTION` "
        "to a one-line description of your retrieval task. They pass through the same validation, seeded query-disjoint "
        "split, floor and baseline, frozen scoring, fine-tuning, held-out evaluation, retrieval, artifact export and "
        "reload-parity cells as the Banking77 sample. The expected schema and the ceilings are stated in the Prerequisites "
        "and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "Qwen3EmbeddingPipeline",
    "guided": {
        "opening": [
            '**Who this notebook is for.** The intended audience is a learner who knows basic Python, has used Colab or Jupyter, and wants to see how a '
            'text-embedding model turns messages into vectors, how retrieval quality is measured against honest floors, and what a bounded contrastive fine-tuning '
            'changes on a real labelled corpus. No prior experience with Qwen3, embeddings or contrastive learning is assumed; terms are explained where they first '
            'matter and again in the **Glossary** at the end. CPU is adequate (about four minutes of model time); a GPU runtime is faster.\n\n**Input → Model → Output.**\n\n| '
            '| Embedding and retrieval | Bounded contrastive fine-tuning |\n|---|---|---|\n| Input | texts: queries (with a task instruction) or documents, up to '
            '8,192 tokens | query–positive pairs: 616 / 154 / 385 Banking77 messages paired with their intent phrase, query-disjoint |\n| Model | the '
            'Qwen3-Embedding-0.6B decoder, last-token pooled and L2-normalised | the same model; only the last two decoder layers train (31.5 M of 595.8 M '
            'parameters) with an InfoNCE loss and in-batch negatives |\n| Output | one unit-length 1024-number vector per text — no score; cosine ranks the 77 '
            'intent documents | recall@1/5/10 and MRR beside a random floor, a lexical baseline and the frozen model, and a 126 MB safetensors adapter that reloads '
            'with parity |\n\n**How to use this notebook.** Choose a runtime (CPU works; **Runtime → Change runtime type → T4 GPU** is faster), then **Runtime → Run '
            "all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are "
            '**infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without '
            'studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the '
            'recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your '
            'reasoning** with a worked answer from the recorded run (the Kaggle Tesla T4 run of 19 September 2026 recorded in `docs/release-verification.md`; on '
            'CUDA the model runs in bfloat16, so your digits may differ slightly). Every adaptation starts from the pinned base, so re-running Section 7 with other '
            'settings is a fresh experiment; Sections 5 and 6 put the pinned base back before they measure, so their numbers are always the frozen '
            'model\'s, and after re-running them you re-run Section 7 before Sections 8 and 9 (those two cells refuse to score the base as "adapted"). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** '
            '1–3 infrastructure → 4 the Banking77 corpus, validation and the split *(evaluation practice)* → 5 the embedding contract *(core concept: vectors, not '
            'scores)* → 6 the random floor, the lexical baseline and the frozen model *(evaluation practice)* → 7 bounded contrastive fine-tuning *(core concept)* '
            '→ 8 held-out evaluation *(evaluation practice)* → 9 retrieval before and after, export and fresh reload *(engineering)* → interpretation, '
            'troubleshooting, glossary and your conclusion.'
        ],
    },
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
        "with no message repeated across splits, and read the near-paraphrases that remain; embed queries and documents through the public API with the instruction contract and read "
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
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available. **Precision differs by device:** the pipeline runs float32 on CPU and bfloat16 on CUDA, so cosine values and the recorded metrics can differ between the two. CPU is adequate for this sample: the build record measured about 5 s to load and digest-verify the 1.19 GB snapshot, about 30 s to embed the 385 test queries and 77 documents, and about 47 s per training epoch over 616 pairs plus a validation pass per epoch. Building the isolated environment (the pinned `torch==2.14.0` among its packages; reused on a re-run) and the 1.19 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and NumPy; what a dense vector, a unit norm and cosine similarity are; what recall@k and mean reciprocal rank measure and why they need a labelled query–document set; what a contrastive (InfoNCE) loss with in-batch negatives does.",
        "- **Data contract:** records are `{id, query, positive}` — a query of 1..100,000 characters (text beyond 8,192 tokens is truncated and flagged at inference; queries and documents are truncated to 64 tokens **during training only**), a positive document of 1..1,000 characters, an optional `negative` document (added to the document set), ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a BYOD dataset needs 12..20,000 distinct queries (20 % go to test and 15 % to validation, and at least 8 must stay for training) and at least 2 distinct positives; queries are de-duplicated ignoring case, punctuation and spacing before splitting so the same message never sits in two splits, and near-paraphrases across splits (token Jaccard ≥ 0.8) are counted and reported, not removed. BYOD accepts CSV, JSON or JSONL in that shape.",
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
                "retrieval candidates, `check_split_disjoint` asserts no message appears in two splits (ignoring case, punctuation "
                "and spacing), `near_duplicate_pairs` counts validation and test messages that are near-paraphrases of a message in "
                "another split (token Jaccard ≥ 0.8) — reported, not refused — and the training "
                "split is written to `outputs/{stem}_train.csv` in the shape BYOD expects. `INSTRUCTION` is the task "
                "description every query carries in later cells.\n\n"
                "Look for: 10,003 + 3,080 raw rows, two digests, splits 616 / 154 / 385, 77 documents, the near-duplicate "
                "count (Banking77 has a few: the same complaint worded twice), and four refusal "
                "probes — a duplicate id, an empty query, a missing field and a dataset too small to split — each rejected "
                "before `torch` does anything.\n\n"
                "*Evaluation practice.* **Predict before running:** customer messages repeat. What would happen to the test numbers if "
                "the same message could sit in both the training and the test split?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n"
                "INSTRUCTION = 'Given a customer support message, retrieve the banking intent it expresses'  # @param {{type:\"string\"}}\n"
                "# A file already in the runtime (works on Colab, Kaggle and Jupyter); empty = the Colab upload dialog.\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.is_file():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} is not a file (relative paths start at {{Path.cwd()}}): give one .csv, .json, .jsonl file holding id, query and positive.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the file in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith(('.csv', '.json', '.jsonl')):\n"
                "            raise ValueError(f'{{file_name}}: upload one .csv, .json, .jsonl file holding id, query and positive.')\n"
                "        byod_path = Path('work') / file_name\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
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
                "# QEM-M3: the 8-record floor is the training floor; validation and test only need one record each.\n"
                "dataset_manifests = {{name: validate_dataset(part) if name == 'train' else validate_dataset(part, min_records=1) for name, part in splits.items()}}\n"
                "document_set = documents([*train_records, *val_records, *test_records])\n"
                "disjoint = check_split_disjoint(splits)\n"
                "near_duplicates = near_duplicate_pairs(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'instruction': INSTRUCTION, 'raw_rows': raw_rows, 'splits': disjoint, 'n_documents': len(document_set), 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_queries': manifest['unique_queries'], 'n_documents': manifest['n_documents'], 'query_chars': manifest['query_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'near_duplicates': near_duplicates['near_duplicates'], 'threshold_jaccard': near_duplicates['threshold'], 'reported_not_removed': True}})\n"
                "for example in near_duplicates['examples'][:2]:\n"
                "    print({{'near_duplicate': example}})\n"
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
                '**What to notice:** 10,003 + 3,080 raw rows, splits 616 / 154 / 385, 77 documents, `unique_queries` per split, the near-duplicate counts, and the four refusals.\n\n<details><summary>Check '
                'your reasoning</summary>They would be inflated: the model would be scored on messages it was trained on, and memorisation would look like skill. '
                "`build_sample_dataset` drops repeated messages and draws training and validation from the release's `train` file and test from its `test` file; "
                '`check_split_disjoint` then asserts no message sits in two splits, ignoring case, punctuation and spacing. That is an exact-repeat check: '
                'paraphrases such as *Has my top-up been cancelled?* / *My top-up has been cancelled.* pass it, and `near_duplicate_pairs` counts them '
                '(9 of the 385 test queries at token Jaccard ≥ 0.8 on this draw) so you can judge their weight — under 3 % here. The refusals (duplicate id, empty query, missing field, too small) stop before '
                '`torch` runs.</details>'
            ),
        },
        {
            "md": (
                "## 5. Embed through the inference contract\n\n"
                "Before any adaptation, the embedding contract is exercised as it always was. `validate_inputs` applies "
                "exactly the checks `embed` applies — both route through the same private `_check_inputs` — so type, "
                "batch size 1..`MAX_BATCH`, non-empty text, the character ceiling, a `kind` in `KINDS` and a non-empty "
                "instruction are enforced identically; it returns an input manifest for all 77 documents (validated in batches of "
                "`MAX_BATCH`, the same batches `embed` receives) with the three "
                "probe queries recorded under `query_manifest`, and a deliberately oversized batch is validated too and its "
                "rejection recorded as a finding. `embed` returns one unit-norm 1024-d vector per text with `n_tokens` and "
                "`truncated` flags; queries carry the instruction prefix, documents do not. The three test queries are "
                "ranked against the 77 documents by cosine and their top-3 documents printed beside the gold intent — a "
                "qualitative look before any metric is read; **cosine is a similarity, not a probability**, and no "
                "threshold ships. The frozen top-3 lists are kept as the *before* column for Section 9. The cell first puts "
                "the pinned base back (`restore_base`) if an earlier Section 7 adapted the model, so a re-run — for example "
                "with your own data — always shows the frozen model here.\n\n"
                "**Predict before running:** the documents are two- or three-word intent phrases such as *card arrival*. Will the frozen "
                "embedder put the right phrase in the top 3 for these three messages?"
            ),
            "code": (
                "import time\n\n"
                "# QEM-M2: this cell describes the frozen model, so an earlier adaptation is undone first.\n"
                "restored_tensors = pipe.restore_base()\n"
                "if restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'an earlier Section 7 had adapted the model; run Section 7 again before Sections 8 and 9'}})\n"
                "probe_records = test_records[:3]\n"
                "probe_queries = [r['query'] for r in probe_records]\n"
                "print({{'ceilings': {{'MAX_BATCH': MAX_BATCH, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'EMBEDDING_DIM': EMBEDDING_DIM, 'MAX_TRAIN_TOKENS': MAX_TRAIN_TOKENS, 'KINDS': KINDS}}}})\n"
                "# QEM-m3: every document is validated, in the same MAX_BATCH batches embed receives, and the manifests merged.\n"
                "document_names = [f'doc-{{i:02d}}' for i in range(len(document_set))]\n"
                "document_manifests = [validate_inputs(document_set[start:start + MAX_BATCH], 'document', names=document_names[start:start + MAX_BATCH]) for start in range(0, len(document_set), MAX_BATCH)]\n"
                "input_manifest = {{**document_manifests[0], 'inputs': [entry for manifest in document_manifests for entry in manifest['inputs']], 'batches': len(document_manifests)}}\n"
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
                "print({{'n_tokens': query_result['n_tokens'], 'seconds': {{'documents': document_seconds, 'queries': query_seconds}}, 'checks': checks, 'findings': len(input_manifest['findings']), 'documents_in_manifest': len(input_manifest['inputs']), 'score_semantics': 'cosine between unit vectors; a similarity, not a probability; no threshold shipped'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** one unit-norm 1024-d vector per text, `checks`, the oversized-batch finding, and each probe's top-3 beside its gold intent.\n\n<details><summary>Check "
                'your reasoning</summary>Often, not always — three queries are an anecdote, and the metric is Section 6. What the cell does establish is the contract: '
                'every vector has norm 1, nothing was truncated, queries carry the instruction and documents do not. A cosine near 0.6 is not "60 % sure": it is a '
                'similarity, and no threshold ships.</details>'
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
                "means. About half a minute on CPU. Like Section 5, the cell puts the pinned base back first, so `frozen_model_test` "
                "is always the pretrained model (`adapted: False`). Whether the frozen model beats the random floor is recorded "
                "as a **verdict**; on your own pairs the run continues either way, while on the Banking77 sample a frozen model "
                "at or below the floor stops the cell, because that would mean a broken run.\n\n"
                "*Evaluation practice.* **Predict before running:** the lexical baseline only counts shared words. How close will it get "
                "to the frozen embedder on recall@1?"
            ),
            "code": (
                "def brief(m):\n"
                "    return {{k: round(m[k], 4) for k in ('recall@1', 'recall@5', 'recall@10', 'mrr')}} | {{'median_rank': m.get('median_rank')}}\n\n"
                "if pipe.restore_base():  # QEM-M2: the frozen numbers are always the pinned base's\n"
                "    print({{'restored_pinned_base': True, 'note': 'an earlier Section 7 had adapted the model; run Section 7 again before Sections 8 and 9'}})\n"
                "floor = random_floor(len(document_set))\n"
                "print({{'random_floor': {{k: round(floor[k], 4) for k in ('recall@1', 'recall@5', 'recall@10', 'mrr')}}, 'baseline': floor['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "baseline_lexical = pipe.lexical_baseline(test_records, document_set)\n"
                "print({{'lexical_baseline': brief(baseline_lexical), 'baseline': baseline_lexical['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "print({{'frozen_model_test': brief(frozen_test), 'n_queries': frozen_test['n_queries'], 'n_documents': frozen_test['n_documents'], 'verdict': frozen_test['verdict'], 'adapted': frozen_test['adapted'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "# Contract integrity: both systems ranked the same document set.\n"
                "if frozen_test['n_documents'] != baseline_lexical['n_documents']:\n"
                "    raise RuntimeError(f\"contract: the frozen model ranked {{frozen_test['n_documents']}} documents, the lexical baseline {{baseline_lexical['n_documents']}}\")\n"
                "# SWP-A: the quality comparison is a recorded verdict, not an assert, so a BYOD run still reaches adaptation and export.\n"
                "frozen_vs_floor = 'above' if frozen_test['mrr'] > floor['mrr'] else 'not above'\n"
                "print({{'verdict_frozen_vs_random_floor_mrr': frozen_vs_floor}})\n"
                "if not USE_BYOD and frozen_vs_floor != 'above':\n"
                "    raise RuntimeError(f\"On the Banking77 sample the frozen model must rank above the random floor (MRR {{frozen_test['mrr']:.4f}} vs {{floor['mrr']:.4f}}): the snapshot or the corpus is not the pinned one. Run all again from the top.\")"
            ),
        },
        {
            "md": (
                '**What to notice:** the three systems on recall@1/5/10 and MRR, `median_rank`, and the verdict line.\n\n<details><summary>Check your '
                'reasoning</summary>Not close. In the recorded run recall@1 was 0.013 for the random floor, 0.3065 for the lexical baseline and 0.6312 for the frozen '
                'embedder (MRR 0.064 / 0.4159 / 0.741; median rank 5 for lexical, 1 for frozen). Shared words such as *card* get the lexical baseline a third of the '
                'way; the frozen embedder already understands paraphrase. A lexical baseline near the model would mean the task is mostly keyword matching.</details>'
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
                "47 s of training plus a validation pass per epoch on CPU). The CPU float32 build record's sweep on this sample (held-out recall@1): two layers at 2e-5 reached 77.1 %, four layers at 2e-5 82.1 % with a 252 MB adapter, two layers at 5e-5 80.3 % with a 126 MB adapter — the default. On a CUDA GPU the model runs in bfloat16: the recorded Kaggle Tesla T4 run of the default reached 76.6 %, so expect the high seventies to low eighties depending on the device. "
                "Every call starts from the pinned base (`started_from` in the printed result), so a re-run with other settings is a "
                "fresh experiment, not continued training, and epoch 0 is always the frozen model.\n\n"
                "**Predict before running:** with a batch of 16, each query's negatives are the other 15 queries' intent phrases. Will "
                "two epochs on 616 pairs be enough to move validation recall@1?"
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 2  # @param {{type:\"integer\"}}\n"
                "TEMPERATURE = 0.05  # @param {{type:\"number\"}}\n"
                "# The recorded defaults; Section 8 holds the Banking77 sample to an improvement only at these settings.\n"
                "DEFAULT_SETTINGS = (EPOCHS, LEARNING_RATE, BATCH_SIZE, TRAINABLE_LAYERS, TEMPERATURE) == (2, 5e-5, 16, 2, 0.05)\n\n"
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
                "print({{'objective': adapt_result['objective'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'train_documents': adapt_result['n_train_documents'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (`note: frozen model`), the training loss, `val_recall@1` and `val_mrr` per epoch, `best_epoch`, and `started_from`.\n\n<details><summary>Check '
                "your reasoning</summary>Yes. Validation recall@1 climbs by about ten points over two epochs while recall@5 passes 95 %; the build record's sweep shows "
                'more layers buy a little more at twice the adapter size. In-batch negatives are cheap — no mining — but they are easy ones; a real corpus with '
                'near-duplicate documents needs harder negatives.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or epoch selection, and no message in it appears in the "
                "training or validation splits. The adapted embedder is scored exactly as the frozen one was in Section 6 "
                "— same queries, same 77 documents, same instruction — and the four rows are put side by side. Look for "
                "recall@1 up by ten points or more and MRR up by about a tenth; the cell records whether the adapted MRR is above the "
                "frozen MRR as a **verdict** (`improved`, `no change` or `worse`) in the report and `result.json` instead of asserting it, "
                "so a run on your own pairs that does not improve still exports and reloads; on the Banking77 sample at the "
                "default settings a result that is not *improved* stops the cell, because the recorded runs all improved. "
                "The cell refuses to run if the pipeline holds the pinned base (Section 5 or 6 re-run after Section 7). 385 queries from one seeded split of one corpus give no dispersion estimate; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain on Banking77 "
                "intents says nothing about your retrieval task until you measure it there.\n\n"
                "*Evaluation practice.* **Predict before running:** which will gain more from fine-tuning — recall@1 or recall@10 — and why?"
            ),
            "code": (
                "if pipe.adapter is None:  # QEM-M2: never score the pinned base as 'adapted'\n"
                "    raise RuntimeError('The pipeline holds the pinned base, not an adapted model (Section 5 or 6 was re-run after Section 7 and put the base back): run Section 7, then this cell.')\n"
                "adapted_test = pipe.evaluate(test_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "adapted_val = pipe.evaluate(val_records, instruction=INSTRUCTION, candidates=documents(val_records))\n"
                "comparison = {{\n"
                "    metric: {{'random_floor': round(floor[metric], 4), 'lexical': round(baseline_lexical[metric], 4), 'frozen': round(frozen_test[metric], 4), 'adapted': round(adapted_test[metric], 4)}}\n"
                "    for metric in ('recall@1', 'recall@5', 'recall@10', 'mrr')\n"
                "}}\n"
                "comparison['median_rank'] = {{'lexical': baseline_lexical['median_rank'], 'frozen': frozen_test['median_rank'], 'adapted': adapted_test['median_rank']}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('recall@1', 'recall@5', 'recall@10', 'mrr')}}\n"
                "delta_mrr = adapted_test['mrr'] - frozen_test['mrr']\n"
                "# SWP-A: the direction is a recorded verdict, not an assert, so a BYOD run always reaches export, reload and result.json.\n"
                "comparison['verdict'] = {{'adapted_vs_frozen_mrr': 'improved' if delta_mrr > 0 else ('no change' if delta_mrr == 0 else 'worse'), 'frozen_vs_random_floor_mrr': frozen_vs_floor}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "if not USE_BYOD and DEFAULT_SETTINGS and comparison['verdict']['adapted_vs_frozen_mrr'] != 'improved':\n"
                "    raise RuntimeError(f\"On the Banking77 sample at the default settings the adapted MRR must be above the frozen MRR ({{adapted_test['mrr']:.4f}} vs {{frozen_test['mrr']:.4f}}): Run all again from the top; if it repeats, report it.\")"
                "\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'instruction': INSTRUCTION,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'near_duplicates': near_duplicates,\n"
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
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the four rows side by side, `delta_vs_frozen`, `median_rank`, and the `verdict`.\n\n<details><summary>Check your '
                'reasoning</summary>Recall@1. In the recorded run recall@1 rose 0.6312 → 0.7662 (+0.135) but recall@10 only 0.9377 → 0.9896 (+0.052): the frozen model '
                'already had the right intent near the top, and fine-tuning mostly moved it to first place. MRR went 0.741 → 0.8498 and the verdict was *improved*. One '
                'seeded split of 385 queries has no dispersion estimate.</details>'
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
                "if pipe.adapter is None:  # QEM-M2: the export must be the model Section 8 evaluated\n"
                "    raise RuntimeError('The pipeline holds the pinned base, not an adapted model (Section 5 or 6 was re-run after Section 7): run Section 7 and Section 8, then this cell.')\n"
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
                "    'near_duplicates': near_duplicates['near_duplicates'],\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'files': {{k: {{'name': v[0], 'bytes': v[1], 'sha256': v[2]}} for k, v in CORPUS_FILES.items()}}, 'license': CORPUS_LICENSE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_queries': probe_queries, 'n_tokens': query_result['n_tokens'], 'seconds': {{'documents': document_seconds, 'queries': query_seconds}}}},\n"
                "    'comparison': comparison,\n"
                "    'verdict': comparison['verdict'],\n"
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
        {
            "md": (
                "**What to notice:** the frozen and adapted top-3 for the three probes, `document_shift`, the artifact's size, and `reload_parity`.\n\n<details><summary>Check "
                'your reasoning</summary>Every vector moves, because the adapter changes layers all inputs share: the self-cosine of each document to its frozen '
                'version is below 1, so cosines from the two models are not comparable. In the recorded run the reloaded adapter gave identical query vectors and the '
                'same MRR on the 77-query parity subset (0.8125 in memory and reloaded).</details>'
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
        "ignoring case, punctuation and spacing), read the near-paraphrase count it reports, and split by user or session when your queries come from one. **Documents:** the document set "
        "here is the label vocabulary; a real retrieval task has many more documents than intents and needs hard "
        "negatives beyond the in-batch ones this contract uses.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled corpus, "
        "validate the demonstrated dataset contract with no message repeated across splits, execute the embedding contract and a bounded "
        "contrastive fine-tuning, evaluate by retrieval metrics against a random floor, a lexical baseline and the frozen "
        "model on an independent split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, embedding quality on any other task, a usable "
        "acceptance threshold, or production fitness.\n\n"
        "## Activity: how much does the second trainable layer buy?\n\n"
        "Run this only after the default Run all has finished; Section 9's exported files stay as they are unless you re-run Section 9.\n\n"
        "1. **Predict.** Write down the held-out recall@1 you expect from the adapted model if only the **last one** decoder layer trains "
        "(half the trainable parameters, a 63 MB adapter). Above or below the default's? By how much?\n"
        "2. **Change.** In Section 7 set `TRAINABLE_LAYERS = 1`; leave every other field as it is.\n"
        "3. **Run.** Run Section 7, then Section 8. Every adaptation starts from the pinned base (`started_from` in Section 7's output), "
        "so this is a fresh experiment, not more training on top of the first run.\n"
        "4. **Observe.** Section 7's epoch 0 (`note: frozen model`) must print the same validation MRR as in the default run — the same "
        "starting point. Then read `delta_vs_frozen` in Section 8 and compare it with the default's.\n"
        "5. **Explain.** Say in one sentence what the second layer added, and whether that is worth twice the adapter size for this task.\n\n"
        "To put the notebook back to the recorded state, set `TRAINABLE_LAYERS = 2` and run Sections 7, 8 and 9 again.\n\n"
        "<details><summary>Check your reasoning</summary>Expect a smaller gain, not none: the frozen model already ranks the right intent near the top, "
        "and one layer can still reorder the last few places, so recall@1 still rises but by less than the default's +13.5 points (Kaggle T4 run). "
        "The exact figure depends on the device; what matters is the direction and the adapter-size trade-off.</details>\n\n"
        "**More experiments (same procedure, they do not affect the default path):** set `TEMPERATURE = 0.2` and read whether the softer "
        "contrast learns as fast; set `EPOCHS = 4` and watch whether validation MRR keeps rising or turns (the best epoch is kept either way); "
        "change `INSTRUCTION` in Section 4 and re-run from Section 5 to re-read the frozen numbers — the model is instruction-aware; or bring "
        "your own pairs through BYOD (re-run from Section 4) and read the lexical baseline before the adapted number. On the Banking77 sample "
        "with changed Section 7 settings, Section 8 reports the verdict instead of stopping.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 4 reports a size or SHA-256 mismatch for a '
        'Banking77 file** — the download from `raw.githubusercontent.com` was cut short or altered. Run Section 4 again; a verified cached file is reused.\n- '
        '**Section 7 is slow** — on CPU each epoch takes about a minute; switch to a T4 GPU for a faster run (bfloat16 there, so digits differ slightly from '
        'the CPU build record).\n- **Your numbers differ slightly from the recorded run** — CUDA runs bfloat16 and CPU float32; the comparison between systems, '
        'not the last digit, is the result.\n- **BYOD: \"CSV is missing columns\" or a record refusal** — the message names the rule: columns `id`, `query`, '
        '`positive` (optional `negative`), at least 12 distinct queries (8 stay for training) and two distinct positives; set `INSTRUCTION` to describe your retrieval task.\n- **BYOD: '
        '"BYOD_PATH … is not a file"** — the path is relative to the working directory printed in the message; give one .csv, .json or .jsonl file.\n- **BYOD: '
        '"the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the file in the runtime and set `BYOD_PATH` to its path.\n- **BYOD: '
        '"Upload exactly one file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **Section 8 or 9 says the pipeline '
        'holds the pinned base** — you re-ran Section 5 or 6 after Section 7, and they put the pretrained weights back so their numbers are the '
        'frozen model\'s. Run Section 7 again, then Section 8 and Section 9.\n\n## Glossary\n\n- **Embedding:** a fixed-length '
        'vector (1024 numbers here) that places a text so similar meanings are close; it carries no score by itself.\n- **Last-token pooling / L2 '
        'normalisation:** the vector is the hidden state of the final token, scaled to length 1 so cosine equals the dot product.\n- **Instruction-aware '
        'query:** queries are prefixed with a one-line task description; documents are embedded as-is.\n- **Cosine similarity:** the angle between two unit '
        'vectors; a similarity, not a probability, and no threshold ships.\n- **Recall@k:** the share of queries whose correct document is among the top k; '
        '**MRR** is the mean of 1 / rank of the correct document.\n- **Random floor / lexical baseline:** what a random ranking achieves; ranking by shared-word '
        '(Jaccard) overlap — what no model at all gets.\n- **InfoNCE with in-batch negatives:** a contrastive loss that asks each query to pick its own positive '
        "among the batch's documents; the other queries' positives are the negatives.\n- **Temperature:** the scale applied to cosines in the loss; lower is a "
        'sharper contrast.\n- **Frozen / adapted / pinned base:** the packaged model; the model after Section 7; the verified packaged weights every adaptation '
        'starts from (`restore_base`).\n- **Query-disjoint split:** no message appears in two splits, ignoring case, punctuation and spacing; near-paraphrases (token Jaccard ≥ 0.8) are counted, not removed.\n- **Isolated environment:** the '
        'separate Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## Conclusion (your notes)\n\nComplete these in your own '
        "words; the recorded run's values are in the **Check your reasoning** answers above.\n\n- On the 385 test queries the frozen embedder reached recall@1 "
        "___ against the lexical baseline's ___ and the random floor's ___.\n- Two epochs of contrastive fine-tuning moved recall@1 to ___ and MRR to ___ "
        '(verdict: ___).\n- The number I would not trust on its own is ___, because ___.\n- Before adapting on my own pairs I would de-duplicate by ___, compare '
        'against ___, and use harder negatives such as ___.\n\n'
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/QwenLM/Qwen3-Embedding\n"
        "- Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models (2025): https://arxiv.org/abs/2506.05176\n"
        "- Efficient Intent Detection with Dual Sentence Encoders (Casanueva et al., 2020; Banking77, CC BY 4.0): https://arxiv.org/abs/2003.04807\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
