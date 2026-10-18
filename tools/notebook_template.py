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
    "notebook_kind": "TUTORIAL",
    "use_notice": '> **⚠️ Notice.** For research, training and evaluation only. Not validated for production, clinical, diagnostic or safety-critical use. Running this notebook downloads third-party model weights and datasets under their own licenses (see the model card) and uses compute on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are responsible for having the right to use it. Results here are tutorial evidence, not benchmarks. Everything is provided "as is", without warranty of any kind. To the maximum extent permitted by law, the maintainers and the DIMER platform accept no liability for its use. An embedding model returns vectors, not judgements: nearness is not relevance or correctness.',
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
        "616 / 154 training and validation pairs and the 385-query exposed development test balanced over the 77 intents from the release's own partition, "
        "builds the pinned 770-query fresh test from the test.csv messages this repository had never scored, "
        "embeds three exposed-development-test queries and the 77 intent documents through the inference contract with an input manifest and a "
        "rejection probe, scores the frozen embedder on the exposed development test and the validation split by recall@1 / recall@5 / recall@10 and MRR over "
        "the 77 documents beside the random floor and a lexical (token-overlap) baseline, runs the pre-registered validation grid "
        "(three learning-rate / layer configurations, selected by validation MRR), runs a bounded contrastive "
        "fine-tuning (InfoNCE with in-batch negatives) of the selected configuration with validation-MRR epoch "
        "selection, scores validation and the exposed development test again and, only under the pre-registered settings, the fresh test, retrieves for the same three queries with the adapted embedder, "
        "exports the adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify "
        "parity, then runs an optional transfer check on CLINC150 (not an MTEB task). The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog "
        "and no configuration edit (NOTEBOOK_SPEC 2.2 §5). On CPU the whole path takes 20–40 minutes of model time "
        "after the downloads (the grid trains three configurations); a T4 took 6.4 minutes for the whole session under recipe 1 and is estimated at 7–12 under the float32 recipe; a CUDA runtime is used automatically when present (float32 on every device since pre-registration amendment 1)."
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
            'matter and again in the **Glossary** at the end. CPU works but is slow (20–40 minutes of model time, most of it the validation grid); a T4 GPU runtime is estimated at 7–12 minutes.\n\n**Input → Model → Output.**\n\n| '
            '| Embedding and retrieval | Bounded contrastive fine-tuning |\n|---|---|---|\n| Input | texts: queries (with a task instruction) or documents, up to '
            '8,192 tokens | query–positive pairs: 616 / 154 Banking77 messages for training and validation, plus a 385-query exposed development test and a 770-query fresh test, each paired with its intent phrase, query-disjoint |\n| Model | the '
            'Qwen3-Embedding-0.6B decoder, last-token pooled and L2-normalised | the same model; only the last two decoder layers train (31.5 M of 595.8 M '
            'parameters) with an InfoNCE loss and in-batch negatives |\n| Output | one unit-length 1024-number vector per text — no score; cosine ranks the 77 '
            'intent documents | recall@1/5/10 and MRR beside a random floor, a lexical baseline and the frozen model, and a 126 MB float32 safetensors adapter that reloads '
            'with parity |\n\n**How to use this notebook.** Choose a runtime (CPU works; **Runtime → Change runtime type → T4 GPU** is faster), then **Runtime → Run '
            "all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are "
            '**infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without '
            'studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults are the '
            'pre-registered settings (`docs/qwen3-embedding-preregistration.md`): changing one makes the run exploratory, and the notebook says so. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your '
            'reasoning** with a worked answer. Numbers in those answers are labelled with their source: the Colab CLI Tesla T4 runs of 9 October 2026 (blob `26a62b80`) and 17 October 2026 (blob `971187e6`, the evaluation of record for '
            'fresh test 1), which agree exactly on the 385 queries now called the exposed development test; both are recorded in `docs/release-verification.md` and were not re-run for '
            'this revision. Both used recipe 1, which trained in bfloat16 on CUDA. This revision trains in float32 on every device (pre-registration amendment 1), so adapted numbers on a GPU will differ, typically upward: on CPU float32 the same configuration reached validation MRR 0.828 against 0.788 on the T4. Every adaptation starts from the pinned base, so re-running Section 7 with other '
            'settings is a fresh experiment; Sections 5 and 6 put the pinned base back before they measure, so their numbers are always the frozen '
            'model\'s, and after re-running them you re-run Section 7 before Sections 8 and 9 (those two cells refuse to score the base as "adapted"). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** '
            '1–3 infrastructure → 4 the Banking77 corpus, validation and the split *(evaluation practice)* → 5 the embedding contract *(core concept: vectors, not '
            'scores)* → 6 the random floor, the lexical baseline and the frozen model *(evaluation practice)* → 7 the pre-registered validation grid and bounded contrastive fine-tuning *(core concept)* '
            '→ 8 evaluation: validation, the exposed development test and the fresh test *(evaluation practice)* → 9 retrieval before and after, export and fresh reload *(engineering)* → 10 an optional transfer check *(evaluation practice)* → interpretation, '
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
        "its intent phrase, so nearest-neighbour retrieval over the 77 documents is intent detection, with documents that are two or three words long.\n\n"
        "**Training-data overlap (DAT9).** Banking77 is an MTEB classification task [6], and Qwen3-Embedding was evaluated on it: the upstream "
        "`evaluation/task_prompts.json` carries a `Banking77Classification` prompt [2]. The technical report [1] neither lists nor excludes Banking77 "
        "from its training mix, so whether this notebook's `train.csv` messages were in the model's training data is undisclosed. Read the frozen "
        "number as possibly in-distribution, not zero-shot, and the adaptation gain as a gain over a model that may already be adapted to this "
        "data. The optional transfer check in Section 10 compares the frozen and adapted models on CLINC150 [5], which is not an MTEB task.\n\n"
        "**Pre-registered evaluation protocol (2026-10-17).** The learning rate and layer count of the old default were compared on the 385 seed-42 "
        "test queries in 2026-09, so those queries are the **exposed development test**: shown every run, not independent evidence. The "
        "configuration is re-checked on validation by a pre-registered rule that keeps the previous default unless a challenger clearly beats it "
        "(Section 7); what makes the reported number independent is the pinned 770-query **fresh test**, which played no part in any choice, drawn from "
        "messages this repository had never scored is read **only under the exact pre-registered settings**; see `docs/release-verification.md` "
        "for the recorded evaluation and its reproductions. Amendment 1 (2026-10-18, `docs/qwen3-embedding-preregistration-amendment-2026-10-18.md`) moved "
        "training to float32 on every device and pre-registered **fresh test 2**; fresh test 1, scored once under the bf16 recipe, is now an exposed test. The carried `metrics.py` ranks the "
        "documents for every held-out query by cosine and reads the rank of its positive: **recall@1**, **recall@5**, "
        "**recall@10** and **MRR**; a **random floor** (1 / 77 recall@1) and a **lexical baseline** (Jaccard token "
        "overlap between message and phrase) frame the frozen number. The fine-tuning question is whether a bounded "
        "contrastive adaptation of the last decoder layers on 616 pairs raises retrieval on messages that are not in this "
        "notebook's training split. Nothing here is a quality claim about your retrieval task: it is one seeded split of one corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, dataset and metrics modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real corpus and validate and split it "
        "with no message repeated across splits, and read the near-paraphrases that remain; embed queries and documents through the public API with the instruction contract and read "
        "the vector contract correctly; read recall@k and MRR beside a random floor and a lexical baseline and "
        "understand what they do and do not measure; select a fine-tuning configuration on validation by a pre-registered rule and "
        "run it with explicit hyperparameters and validation-based epoch selection; tell an exposed development test from a fresh "
        "test scored only under the pre-registered settings; compare retrieved "
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
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available. **Precision:** since pre-registration amendment 1 (2026-10-18) the pipeline loads, trains and evaluates in float32 on CPU and CUDA alike (recipe 2); CPU and GPU kernels still differ in low-order digits. Recipe 1, recorded on 17 October 2026, trained in bfloat16 on CUDA, and its adapted numbers were 0.03–0.10 MRR below the CPU float32 ones. CPU is adequate for this sample: the build record measured about 5 s to load and digest-verify the 1.19 GB snapshot, about 30 s to embed the 385 test queries and 77 documents, and, in the 2026-10-17 CPU build record, about 7–9 minutes per grid configuration (two epochs over 616 pairs, each with a validation pass; the default path trains four times: three grid configurations and the selected one). Building the isolated environment (the pinned `torch==2.14.0` among its packages; reused on a re-run) and the 1.19 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and NumPy; what a dense vector, a unit norm and cosine similarity are; what recall@k and mean reciprocal rank measure and why they need a labelled query–document set; what a contrastive (InfoNCE) loss with in-batch negatives does.",
        "- **Data contract:** records are `{id, query, positive}` — a query of 1..100,000 characters (text beyond 8,192 tokens is truncated and flagged at inference; queries and documents are truncated to 64 tokens **during training only**), a positive document of 1..1,000 characters, an optional `negative` document (added to the document set), ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a BYOD dataset needs 12..20,000 distinct queries (20 % go to test and 15 % to validation, and at least 8 must stay for training) and at least 2 distinct positives; queries are de-duplicated ignoring case, punctuation and spacing before splitting so the same message never sits in two splits, and near-paraphrases across splits (token Jaccard ≥ 0.8) are counted and reported, not removed. BYOD accepts CSV, JSON or JSONL in that shape.",
        "- **Validation is structural, not semantic:** nothing checks that a positive is relevant to its query or that the instruction describes the task — a mislabelled pair set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — an internal query log with its relevance labels is exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches two pinned objects (`train.csv` 839,073 bytes, `test.csv` 239,961 bytes; SHA-256 `b06e26ac…` / `d12d6e3b…`) from `raw.githubusercontent.com` at the pinned `PolyAI-LDN/task-specific-datasets` commit over HTTPS, each refused on any mismatch before it is read; Banking77 is CC BY 4.0 (Casanueva et al., 2020). The optional transfer check (Section 10, on by default) fetches one more pinned object, CLINC150 `data_full.json` (2,495,390 bytes, SHA-256 `36923c37…`, CC BY 3.0), from the pinned `clinc/oos-eval` commit.",
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
                "intents. Those 385 test pairs were used to compare configurations in 2026-09, so they are the **exposed development "
                "test**. `build_fresh_test_2` then draws the pinned **fresh test 2** (pre-registration amendment 1): 10 per intent from the `test` messages no run had "
                "never scored (not the 385, not the 154 of the semantic-search workshop, and no exact repeat of a `train` message), refused "
                "unless its manifest SHA-256 matches the pin; fresh test 1, scored once on 2026-10-17 under the bf16 recipe, is excluded too and is now an exposed test. It is built here and not scored until Section 8, and only under the pre-registered settings. `validate_dataset` then checks every record against the contract, `documents` lists the 77 "
                "retrieval candidates, `check_split_disjoint` asserts no message appears in two splits (ignoring case, punctuation "
                "and spacing), `near_duplicate_pairs` counts validation and test messages that are near-paraphrases of a message in "
                "another split (token Jaccard ≥ 0.8) — reported, not refused — and the training "
                "split is written to `outputs/{stem}_train.csv` in the shape BYOD expects. `INSTRUCTION` is the task "
                "description every query carries in later cells.\n\n"
                "Look for: 10,003 + 3,080 raw rows, two digests, splits 616 / 154 / 385, 77 documents, the fresh-test manifest (770 "
                "queries; 1,308 scored or excluded messages left out: the 516 exposed, the 22 exact repeats of training messages and fresh test 1's 770), the near-duplicate "
                "counts (Banking77 has a few: the same complaint worded twice), and four refusal "
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
                "# DAT20: re-running this cell starts a new run; an earlier split is never left for later cells to read.\n"
                "for _stale in ('splits', 'train_records', 'val_records', 'exposed_records', 'report_records', 'fresh', 'fresh_records', 'fresh1_records', 'grid_selection', 'adapt_result', 'run_settings', 'preregistered'):\n"
                "    globals().pop(_stale, None)\n"
                "if USE_BYOD:\n"
                "    try:\n"
                "        if BYOD_PATH.strip():\n"
                "            byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "            if not byod_path.is_file():\n"
                "                raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} is not a file (relative paths start at {{Path.cwd()}}): give one .csv, .json, .jsonl file holding id, query and positive.')\n"
                "            file_name = byod_path.name\n"
                "        else:\n"
                "            try:\n"
                "                from google.colab import files\n"
                "            except ImportError:\n"
                "                raise ValueError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the file in the runtime and set BYOD_PATH to its path.') from None\n"
                "            uploaded = files.upload() or {{}}\n"
                "            if len(uploaded) != 1:\n"
                "                raise ValueError(f'Upload exactly one file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "            file_name, payload = next(iter(uploaded.items()))\n"
                "            if not file_name.lower().endswith(('.csv', '.json', '.jsonl')):\n"
                "                raise ValueError(f'{{file_name}}: upload one .csv, .json, .jsonl file holding id, query and positive.')\n"
                "            byod_path = Path('work') / file_name\n"
                "            byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "            byod_path.write_bytes(payload)\n"
                "        records = load_byod_dataset(byod_path)\n"
                "        splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "        for _name, _part in splits.items():\n"
                "            validate_dataset(_part) if _name == 'train' else validate_dataset(_part, min_records=1)\n"
                "    except (OSError, TypeError, ValueError, KeyError) as exc:\n"
                "        raise DatasetRefused(f'BYOD refused: {{str(exc).rstrip(\".\")}}. Fix the file named here and run this cell again. Files already in outputs/ come from an earlier run and are not this run\\'s result.') from None\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/banking77'))\n"
                "    raw_rows = {{name: len(part) for name, part in corpus.items()}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "train_records, val_records, exposed_records = splits['train'], splits['validation'], splits['test']\n"
                "# The exposed development test is defined for the Banking77 seed-42 draw only; elsewhere Sections 5, 6 and 8 report validation.\n"
                "report_on_exposed = not USE_BYOD and SPLIT_SEED == SAMPLE_SEED\n"
                "report_records = exposed_records if report_on_exposed else val_records\n"
                "report_label = 'exposed development test' if report_on_exposed else 'validation (no exposed development test for this data)'\n"
                "if not USE_BYOD:\n"
                "    fresh = build_fresh_test_2(corpus, splits)\n"
                "    fresh_records = fresh['records']\n"
                "    fresh1_records = fresh['fresh_test_1']\n"
                "    check_split_disjoint({{'train': train_records, 'validation': val_records, 'fresh_test': fresh_records}})\n"
                "else:\n"
                "    fresh, fresh_records, fresh1_records = None, [], []\n"

                "# QEM-M3: the 8-record floor is the training floor; validation and test only need one record each.\n"
                "dataset_manifests = {{name: validate_dataset(part) if name == 'train' else validate_dataset(part, min_records=1) for name, part in splits.items()}}\n"
                "document_set = documents([*train_records, *val_records, *exposed_records])\n"
                "disjoint = check_split_disjoint(splits)\n"
                "near_duplicates = near_duplicate_pairs(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'instruction': INSTRUCTION, 'raw_rows': raw_rows, 'splits': disjoint, 'n_documents': len(document_set), 'file_sha256': {{k: v[2][:12] + '...' for k, v in CORPUS_FILES.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'unique_queries': manifest['unique_queries'], 'n_documents': manifest['n_documents'], 'query_chars': manifest['query_chars'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "print({{'near_duplicates': near_duplicates['near_duplicates'], 'threshold_jaccard': near_duplicates['threshold'], 'reported_not_removed': True}})\n"
                "if fresh is not None:\n"
                "    print({{'fresh_test': {{k: v for k, v in fresh['manifest'].items() if k not in ('near_duplicates_vs_train_validation', 'source')}}, 'scored': 'only under the pre-registered settings, in Section 8'}})\n"
                "    print({{'fresh_test_near_duplicates': {{k: v for k, v in fresh['manifest']['near_duplicates_vs_train_validation'].items() if k != 'examples'}}}})\n"
                "print({{'reported_split': report_label}})\n"

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
                '(9 of the 385 exposed-development-test queries at token Jaccard ≥ 0.8 on this draw; the fresh test reports its own count) so you can judge their weight — under 3 % here. The refusals (duplicate id, empty query, missing field, too small) stop before '
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
                "`truncated` flags; queries carry the instruction prefix, documents do not. Three queries of the reported split (the exposed development test on the default path) are "
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
                "probe_records = report_records[:3]\n"
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
                "## 6. The random floor, the lexical baseline and the frozen model on the exposed development test\n\n"
                "Three numbers frame the adaptation, all over the same 77-document set. The **random floor** is what a "
                "uniformly random ranking achieves in expectation (recall@1 = 1 / 77, MRR ≈ 0.064). The **lexical "
                "baseline** ranks the documents for each query by Jaccard overlap of lower-cased tokens — what a system "
                "with no model gets from shared words such as *card* or *pin*. `pipe.evaluate` embeds the 77 documents "
                "once and every exposed-development-test query with `INSTRUCTION`, ranks by cosine, and reads the rank of each query's own "
                "positive; ties are counted against the positive. Look for the frozen embedder well above both — the "
                "previous blob's recorded runs saw recall@1 in the sixties and MRR in the seventies on these queries — and read `median_rank` beside the "
                "means. About half a minute on CPU. Like Section 5, the cell puts the pinned base back first, so `frozen_model_test` "
                "is always the pretrained model (`adapted: False`). The frozen model is also scored on the 154 validation queries. Whether it beats the random floor **on validation** is recorded "
                "as a **verdict** (a stop must never depend on a test number); on your own pairs the run continues either way, while on the Banking77 sample a frozen model "
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
                "baseline_lexical = pipe.lexical_baseline(report_records, document_set)\n"
                "print({{'lexical_baseline': brief(baseline_lexical), 'baseline': baseline_lexical['baseline'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_report = pipe.evaluate(report_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "print({{'frozen_model': brief(frozen_report), 'split': report_label, 'n_queries': frozen_report['n_queries'], 'n_documents': frozen_report['n_documents'], 'verdict': frozen_report['verdict'], 'adapted': frozen_report['adapted'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "val_documents = documents(val_records)\n"
                "frozen_val = pipe.evaluate(val_records, instruction=INSTRUCTION, candidates=val_documents)\n"
                "val_floor = random_floor(len(val_documents))\n"
                "print({{'frozen_model_validation': brief(frozen_val), 'random_floor_validation_mrr': round(val_floor['mrr'], 4)}})\n"
                "print({{'definitions': frozen_report['definitions']}})\n"

                "# Contract integrity: both systems ranked the same document set.\n"
                "if frozen_report['n_documents'] != baseline_lexical['n_documents']:\n"
                "    raise RuntimeError(f\"contract: the frozen model ranked {{frozen_report['n_documents']}} documents, the lexical baseline {{baseline_lexical['n_documents']}}\")\n"
                "# SWP-A: the quality comparison is a recorded verdict, not an assert, so a BYOD run still reaches adaptation and export.\n"
                "# DOC11: the stop reads validation, never a test split.\n"
                "frozen_vs_floor = 'above' if frozen_val['mrr'] > val_floor['mrr'] else 'not above'\n"
                "print({{'verdict_frozen_vs_random_floor_mrr': frozen_vs_floor}})\n"
                "if not USE_BYOD and frozen_vs_floor != 'above':\n"
                "    raise RuntimeError(f\"On the Banking77 sample the frozen model must rank above the random floor on validation (MRR {{frozen_val['mrr']:.4f}} vs {{val_floor['mrr']:.4f}}): the snapshot or the corpus is not the pinned one. Run all again from the top.\")"
            ),
        },
        {
            "md": (
                '**What to notice:** the three systems on recall@1/5/10 and MRR, `median_rank`, and the verdict line.\n\n<details><summary>Check your '
                'reasoning</summary>Not close. On these 385 queries, in the Colab CLI T4 runs of 9 and 17 October 2026 (blobs `26a62b80` and `971187e6`, identical; recipe 1, bfloat16 training on CUDA; not re-run for this revision), recall@1 was 0.013 for the random floor, 0.3065 for the lexical baseline and 0.6312 for the frozen '
                'embedder (MRR 0.064 / 0.4159 / 0.741; median rank 5 for lexical, 1 for frozen). Shared words such as *card* get the lexical baseline a third of the '
                'way; the frozen embedder already understands paraphrase. A lexical baseline near the model would mean the task is mostly keyword matching.</details>'
            ),
        },
        {
            "md": (
                "## 7. Bounded contrastive fine-tuning, configured on validation\n\n**7a. The pre-registered validation grid.** The learning rate and the number of trainable layers are chosen here, on the 154 validation queries only, by a rule written down before any run (`docs/qwen3-embedding-preregistration.md`). Three configurations are trained with every other setting fixed (2 epochs, batch 16, temperature 0.05, seed 0): **C0** 2 layers at 5e-5 (the previous default), **C1** 2 layers at 2e-5 and **C2** 4 layers at 2e-5. Each is scored by its best-epoch validation MRR. `select_configuration` keeps C0 unless a challenger beats it by at least **0.02** (a predeclared practical tolerance, not a significance test); C2, which doubles the adapter, must also beat the best 2-layer configuration by 0.02; and the frozen model is kept unless the selected configuration beats the frozen validation MRR by 0.02. Every configuration starts from the pinned base, and the grid never reads a test split. It took 139 s on a T4 under recipe 1 (bfloat16; estimated 2.5–5 minutes in float32) and about 25 minutes on a workstation CPU (2026-10-17 build record), and it is on by default (`RUN_GRID`). With `RUN_GRID = False` the training cell uses C0, and the run is exploratory because no selection was made in this session.\n\n**Predict before running:** every configuration gets the same budget, 2 epochs (78 optimiser steps). With a learning rate two and a half times C1's, will C0 get further within that budget, and will either challenger clear a 0.02 margin on validation MRR?"
            ),
            "code": (
                "RUN_GRID = True  # @param {{type:\"boolean\"}}\n"
                "\n"
                "def grid_report(entry):\n"
                "    if entry.get('val'):\n"
                "        print({{'epoch': entry['epoch'], 'val_recall@1': round(entry['val']['recall@1'], 4), 'val_mrr': round(entry['val']['mrr'], 4)}})\n"
                "\n"
                "grid_selection = None\n"
                "if RUN_GRID:\n"
                "    grid_rows = {{}}\n"
                "    t0 = time.perf_counter()\n"
                "    for name, config in GRID_CONFIGS.items():\n"
                "        print({{'config': name, **config}})\n"
                "        result = pipe.adapt(train_records, val_records, instruction=INSTRUCTION, epochs=PREREGISTERED_SETTINGS['epochs'], lr=config['lr'], batch_size=PREREGISTERED_SETTINGS['batch_size'], trainable_layers=config['trainable_layers'], temperature=PREREGISTERED_SETTINGS['temperature'], seed=PREREGISTERED_SETTINGS['adapt_seed'], progress=grid_report)\n"
                "        best = result['history'][result['best_epoch']]['val']\n"
                "        grid_rows[name] = {{**config, 'best_epoch': result['best_epoch'], 'val_recall@1': best['recall@1'], 'val_mrr': best['mrr'], 'frozen_val_mrr': result['history'][0]['val']['mrr']}}\n"
                "    pipe.restore_base()\n"
                "    grid_seconds = round(time.perf_counter() - t0, 1)\n"
                "    grid_selection = select_configuration({{name: row['val_mrr'] for name, row in grid_rows.items()}}, grid_rows['C0']['frozen_val_mrr'])\n"
                "    for name, row in grid_rows.items():\n"
                "        print({{name: {{k: round(v, 4) if isinstance(v, float) and k != 'lr' else v for k, v in row.items()}}}})\n"
                "    print({{'selection': {{k: grid_selection[k] for k in ('config', 'settings', 'reason', 'selected_model')}}, 'margin': grid_selection['margin'], 'seconds': grid_seconds}})\n"
                "    # SCORE_ONCE item 5: each grid run has its own record; the first pre-registered one is never overwritten.\n"
                "    grid_runs = globals().get('grid_runs', 0) + 1\n"
                "    grid_data_preregistered = report_on_exposed and INSTRUCTION == DEFAULT_INSTRUCTION\n"
                "    first_grid = globals().get('first_grid_selection')\n"
                "    if grid_data_preregistered and first_grid is None:\n"
                "        grid_tag, first_grid_selection = '', grid_selection\n"
                "    elif grid_data_preregistered and first_grid['config'] == grid_selection['config']:\n"
                "        grid_tag = f'_reproduction_{{grid_runs}}'\n"
                "    else:\n"
                "        grid_tag = f'_exploratory_{{grid_runs}}'\n"
                "        if grid_data_preregistered:\n"
                "            print({{'grid_selection_changed': {{'first': first_grid['config'], 'now': grid_selection['config']}}, 'consequence': 'recorded as exploratory; the fresh test is not read for a selection that differs from the first pre-registered grid of this runtime'}})\n"
                "    with open(f'outputs/{stem}_grid_validation{{grid_tag}}.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'test_look': 'not scored (selection-only)', 'grid_run': grid_runs, 'grid_tag': grid_tag, 'data_source': data_source, 'instruction': INSTRUCTION, 'rows': grid_rows, 'selection': grid_selection, 'seconds': grid_seconds, 'repository_revision': NOTEBOOK_SOURCE['repository_revision']}}, handle, indent=2, ensure_ascii=False)\n"
                "else:\n"
                "    print({{'grid': 'not run', 'consequence': 'the training cell uses C0 and this run is exploratory (no pre-registered selection in this session)'}})"
            ),
        },
        {
            "md": (
                '**What to notice:** each configuration\'s epoch 0 (the same frozen validation MRR every time), its best epoch, the three best-epoch validation MRRs side by side, and which rule decided the `selection`.\n\n<details><summary>Check your reasoning</summary>The grid prints the answer for this run; the recorded hosted grid (T4, 17 October 2026, blob `971187e6`, recipe 1 in bfloat16) gave validation MRR C0 0.7877, C1 0.7197 and C2 0.7299 against frozen 0.6962 and selected C0; the CPU float32 build record gave 0.8284 / 0.7962 / 0.8270 (frozen 0.6991) and also selected C0. Recipe 2 has no recorded grid yet. What matters is the procedure: the choice is made on validation by a rule fixed in advance, and a difference smaller than the margin keeps the smaller, current configuration rather than chasing noise. In both records every configuration\'s best epoch was the last one (epoch 2): '
                'validation MRR was still rising when the budget ran out, so the grid compares configurations **at this step budget**, not learning rates in general, and a '
                'lower learning rate is handicapped by construction. The `EPOCHS = 4` experiment (exploratory) shows whether C1 catches up.</details>'
            ),
        },
        {
            "md": (
                "### 7b. Train the selected configuration\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_LAYERS` decoder layers — two for C0 and C1 "
                "(31,461,888 of 595,776,512 parameters), four for C2; the token embeddings, the earlier layers and the final norm stay "
                "frozen — with an **InfoNCE** loss: each batch embeds its queries (with `INSTRUCTION`) and the unique "
                "documents among its positives in one forward pass, and every query must pick its own positive out of the "
                "batch's documents by cosine at `TEMPERATURE` — the other queries' positives are the negatives. AdamW at a "
                "fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler; texts are truncated to "
                "`MAX_TRAIN_TOKENS` (64) **during training only**. Epoch 0 records the frozen model's validation retrieval "
                "metrics over the validation document set; every epoch is scored the same way, and the epoch with the "
                "highest validation MRR is kept.\n\n"
                "Watch validation recall@1 climb by ten points or so over two epochs while recall@5 passes 95 % ("
                "the previous blob's recorded runs; training the selected configuration takes about 7–9 minutes on a workstation CPU in the 2026-10-17 build record and took 47.7 s on a T4 under recipe 1). `LEARNING_RATE = 0` and `TRAINABLE_LAYERS = 0` "
                "mean *use the grid's selection*; any other value, like any other change to these fields, makes the run exploratory, and Section 8 "
                "then reads validation and the exposed development test only. The cell prints `preregistered` and the `settings_differing` list. "
                "Every call starts from the pinned base (`started_from` in the printed result), so a re-run with other settings is a "
                "fresh experiment, not continued training, and epoch 0 is always the frozen model.\n\n"
                "**Predict before running:** with a batch of 16, each query's negatives are the other 15 queries' intent phrases. Will "
                "two epochs on 616 pairs be enough to move validation recall@1?"
            ),
            "code": (
                "EPOCHS = 2  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 0.0  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 16  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_LAYERS = 0  # @param {{type:\"integer\"}}\n"
                "TEMPERATURE = 0.05  # @param {{type:\"number\"}}\n"
                "ADAPT_SEED = 0\n"
                "# A Section 7b that does not finish (for example an out-of-range field) leaves no pre-registered state behind.\n"
                "preregistered, settings_differing = False, ['section 7b did not finish']\n"
                "globals().pop('run_settings', None)\n"
                "# 0 = the configuration the Section 7a grid selected (C0 when the grid did not run in this session).\n"
                "chosen = grid_selection['settings'] if globals().get('grid_selection') else GRID_CONFIGS['C0']\n"
                "lr = LEARNING_RATE or chosen['lr']\n"
                "layers = TRAINABLE_LAYERS or chosen['trainable_layers']\n\n"
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
                "adapt_result = pipe.adapt(train_records, val_records, instruction=INSTRUCTION, epochs=EPOCHS, lr=lr, batch_size=BATCH_SIZE, trainable_layers=layers, temperature=TEMPERATURE, seed=ADAPT_SEED, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'objective': adapt_result['objective'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'train_documents': adapt_result['n_train_documents'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})\n"
                "# SCORE_ONCE: `preregistered` is computed from every pre-registered setting, never from one field.\n"
                "run_settings = {{\n"
                "    'data_source': PREREGISTERED_SETTINGS['data_source'] if not USE_BYOD else data_source,\n"
                "    'split_seed': SPLIT_SEED,\n"
                "    'sizes': {{'train': len(train_records), 'validation': len(val_records), 'exposed_development_test': len(exposed_records), 'fresh_test': len(fresh_records)}},\n"
                "    'fresh_test_sha256': fresh['manifest']['manifest_sha256'] if fresh else None,\n"
                "    'instruction': INSTRUCTION,\n"
                "    'epochs': EPOCHS,\n"
                "    'batch_size': BATCH_SIZE,\n"
                "    'temperature': TEMPERATURE,\n"
                "    'max_train_tokens': adapt_result['max_train_tokens'],\n"
                "    'adapt_seed': ADAPT_SEED,\n"
                "    'precision': adapt_result['precision'],\n"
                "    'optimizer': PREREGISTERED_SETTINGS['optimizer'],\n"
                "    'epoch_selection': adapt_result['selection'],\n"
                "    'grid': PREREGISTERED_SETTINGS['grid'] if globals().get('grid_selection') else 'not run',\n"
                "    'configuration': {{'trainable_layers': layers, 'lr': lr}},\n"
                "}}\n"
                "settings_differing = preregistration_diff(run_settings, globals().get('grid_selection'))\n"
                "preregistered = not settings_differing\n"
                "print({{'configuration': run_settings['configuration'], 'preregistered': preregistered, 'settings_differing': settings_differing}})\n"
                "if not preregistered:\n"
                "    print({{'exploratory_run': 'Section 8 will read validation and the exposed development test only; the fresh test is not scored for this run'}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (`note: frozen model`), the training loss, `val_recall@1` and `val_mrr` per epoch, `best_epoch`, and `started_from`.\n\n<details><summary>Check '
                "your reasoning</summary>Yes. In the previous blob's recorded runs validation recall@1 climbed by about ten points over two epochs while recall@5 passed 95 %; the grid's C2 row shows "
                'what two more layers buy on validation at twice the adapter size. In-batch negatives are cheap — no mining — but they are easy ones; a real corpus with '
                'near-duplicate documents needs harder negatives.</details>'
            ),
        },
        {
            "md": (
                "## 8. Evaluation: validation, the exposed development test and the fresh test\n\n"
                "**Pre-registered evaluation protocol (2026-10-17).** No message in any evaluated split repeats a training message (ignoring case, punctuation and spacing). The adapted "
                "embedder is scored exactly as the frozen one was in Section 6 (same queries, same 77 documents, same instruction) on validation "
                "and on the **exposed development test**, which is shown every run for continuity and is not independent evidence. The **fresh "
                "test** (since amendment 1, **fresh test 2**, 770 queries) is scored **only under the exact pre-registered settings**: `preregistered` comes from every one of them, "
                "including the grid's selection. Any other setting (a changed form field, your own data, the Activity) is exploratory: the cell "
                "says so, reads validation and the exposed development test only, and writes `*_exploratory_<n>` files instead of the "
                "pre-registered report. A pre-registered scoring is labelled `pre-registered evaluation (this runtime)` the first time in a runtime "
                "and `reproduction` after that; see `docs/release-verification.md` for the recorded evaluation and its reproductions. On the fresh "
                "test the cell prints the paired comparison: each query's reciprocal rank under both models, the MRR difference, how many queries "
                "improved or got worse, and a paired bootstrap 95 % interval. The stop that keeps a broken run from exporting reads **validation**: on "
                "the Banking77 sample under the pre-registered settings an adapted validation MRR that is not above the frozen one stops the cell; "
                "results on the fresh test are verdicts only, never stops. **Fresh test 1** (scored once, 2026-10-17, bf16 recipe; now an exposed test) is shown "
                "every run for continuity, like the exposed development test, and is never a selection input or a stop. When the pre-registered rule kept the frozen model (`keep_frozen`), that stop does "
                "not apply: the frozen model is the system scored on the fresh test, and Section 9 writes the rejected adapter under a `_not_selected` name. "
                "The adapted validation numbers were used to choose the epoch and the configuration, so they are optimistic; the fresh test is the "
                "unbiased comparison. The fresh test has 10 queries per intent, and queries of one intent share a document, so beside the query-level "
                "interval (the pre-registered one) the cell prints an intent-level interval that resamples whole intents, a disclosed second analysis "
                "that is usually wider. "
                "The cell refuses to run if the pipeline holds the pinned base (Section 5 or 6 re-run after Section 7). The bootstrap interval covers query sampling only, not seed-to-seed variation, and everything here is one seeded split of one corpus; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain on Banking77 "
                "intents says nothing about your retrieval task until you measure it there.\n\n"
                "*Evaluation practice.* **Predict before running:** which will gain more from fine-tuning — recall@1 or recall@10 — and why?"
            ),
            "code": (
                "if pipe.adapter is None:  # QEM-M2: never score the pinned base as 'adapted'\n"
                "    raise RuntimeError('The pipeline holds the pinned base, not an adapted model (Section 5 or 6 was re-run after Section 7 and put the base back): run Section 7, then this cell.')\n"
                "adapted_report = pipe.evaluate(report_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "adapted_val = pipe.evaluate(val_records, instruction=INSTRUCTION, candidates=val_documents)\n"
                "comparison = {{\n"
                "    metric: {{'random_floor': round(floor[metric], 4), 'lexical': round(baseline_lexical[metric], 4), 'frozen': round(frozen_report[metric], 4), 'adapted': round(adapted_report[metric], 4)}}\n"
                "    for metric in ('recall@1', 'recall@5', 'recall@10', 'mrr')\n"
                "}}\n"
                "comparison['split'] = report_label\n"
                "comparison['median_rank'] = {{'lexical': baseline_lexical['median_rank'], 'frozen': frozen_report['median_rank'], 'adapted': adapted_report['median_rank']}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_report[metric] - frozen_report[metric], 4) for metric in ('recall@1', 'recall@5', 'recall@10', 'mrr')}}\n"
                "comparison['validation_mrr'] = {{'frozen': round(frozen_val['mrr'], 4), 'adapted': round(adapted_val['mrr'], 4)}}\n"
                "\n"
                "def direction(delta):\n"
                "    return 'improved' if delta > 0 else ('no change' if delta == 0 else 'worse')\n"
                "\n"
                "delta_mrr = adapted_report['mrr'] - frozen_report['mrr']\n"
                "delta_val_mrr = adapted_val['mrr'] - frozen_val['mrr']\n"
                "# SWP-A: the directions are recorded verdicts, so a BYOD or exploratory run always reaches export, reload and result.json.\n"
                "comparison['verdict'] = {{'adapted_vs_frozen_mrr': direction(delta_mrr), 'adapted_vs_frozen_validation_mrr': direction(delta_val_mrr), 'frozen_vs_random_floor_validation_mrr': frozen_vs_floor}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "# DOC11: the stop reads validation, never a test split.\n"
                "# F5: a selection that differs from the first pre-registered grid of this runtime is never a reproduction.\n"
                "first_grid = globals().get('first_grid_selection')\n"
                "if preregistered and first_grid is not None and grid_selection['config'] != first_grid['config']:\n"
                "    preregistered, settings_differing = False, [*settings_differing, 'configuration differs from the first pre-registered grid of this runtime']\n"
                "# The pre-registered keep-frozen rule: when the grid kept the frozen model, the adapted-above-frozen stop does not apply.\n"
                "keep_frozen = bool(preregistered and grid_selection['keep_frozen'])\n"
                "if not USE_BYOD and preregistered and not keep_frozen and comparison['verdict']['adapted_vs_frozen_validation_mrr'] != 'improved':\n"
                "    raise RuntimeError(f\"On the Banking77 sample under the pre-registered settings the adapted validation MRR must be above the frozen validation MRR ({{adapted_val['mrr']:.4f}} vs {{frozen_val['mrr']:.4f}}): Run all again from the top; if it repeats, report it.\")\n"
                "\n"
                "# SCORE_ONCE: the fresh test is read only under the pre-registered settings.\n"
                "fresh_looks = globals().get('fresh_looks', 0)\n"
                "exploratory_runs = globals().get('exploratory_runs', 0)\n"
                "look = test_look(preregistered, fresh_looks)\n"
                "fresh_result = None\n"
                "if preregistered:\n"
                "    frozen_fresh = pipe.evaluate_base(fresh_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "    adapted_fresh = None if keep_frozen else pipe.evaluate(fresh_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "    fresh_result = {{\n"
                "        'test_look': look,\n"
                "        'selected_model': grid_selection['selected_model'],\n"
                "        'n_queries': frozen_fresh['n_queries'],\n"
                "        'frozen': brief(frozen_fresh),\n"
                "        'adapted': brief(adapted_fresh) if adapted_fresh else None,\n"
                "        'paired_mrr': paired_rank_comparison(frozen_fresh['ranks'], adapted_fresh['ranks']) if adapted_fresh else None,\n"
                "        'paired_mrr_intent_cluster': paired_cluster_comparison(frozen_fresh['ranks'], adapted_fresh['ranks'], [r['intent'] for r in fresh_records]) if adapted_fresh else None,\n"
                "        'manifest_sha256': fresh['manifest']['manifest_sha256'],\n"
                "        'device': str(pipe.device),\n"
                "        'dtype': MODEL_DTYPE,\n"
                "    }}\n"
                "    if adapted_fresh:\n"
                "        fresh_result['verdict'] = direction(adapted_fresh['mrr'] - frozen_fresh['mrr'])\n"
                "    fresh_looks += 1\n"
                "    run_tag = '' if fresh_looks == 1 and look != 'reproduction' else f'_reproduction_{{fresh_looks}}'\n"
                "    print({{'fresh_test': {{k: v for k, v in fresh_result.items() if k not in ('paired_mrr', 'paired_mrr_intent_cluster')}}}})\n"
                "    if fresh_result['paired_mrr']:\n"
                "        paired = fresh_result['paired_mrr']\n"
                "        print({{'fresh_test_paired_mrr': {{'delta': round(paired['delta_mrr'], 4), 'ci95': [round(x, 4) for x in paired['ci']], 'improved': paired['n_improved'], 'worse': paired['n_worse'], 'unchanged': paired['n_unchanged']}}, 'ci95_intent_level': [round(x, 4) for x in fresh_result['paired_mrr_intent_cluster']['ci']]}})\n"
                "    print({{'test_look': look, 'note': 'a runtime cannot know about other runtimes; docs/release-verification.md names the evaluation of record'}})\n"
                "else:\n"
                "    exploratory_runs += 1\n"
                "    run_tag = f'_exploratory_{{exploratory_runs}}'\n"
                "    print({{'test_look': look, 'settings_differing': settings_differing, 'note': 'exploratory: validation and the exposed development test only; the fresh test was not read'}})\n"
                "# Fresh test 1 (scored once, 2026-10-17, bf16 recipe) is now an exposed test: shown for continuity, never a selection input or a stop.\n"
                "fresh1_result = None\n"
                "if fresh1_records:\n"
                "    f1_frozen = pipe.evaluate_base(fresh1_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "    f1_adapted = pipe.evaluate(fresh1_records, instruction=INSTRUCTION, candidates=document_set)\n"
                "    fresh1_result = {{'label': FRESH_TEST_1_LABEL, 'n_queries': f1_frozen['n_queries'], 'frozen': brief(f1_frozen), 'adapted': brief(f1_adapted), 'paired_mrr': paired_rank_comparison(f1_frozen['ranks'], f1_adapted['ranks'])}}\n"
                "    print({{'fresh_test_1': {{'label': FRESH_TEST_1_LABEL, 'frozen': brief(f1_frozen), 'adapted': brief(f1_adapted), 'delta_mrr': round(fresh1_result['paired_mrr']['delta_mrr'], 4), 'ci95_queries': [round(x, 4) for x in fresh1_result['paired_mrr']['ci']]}}}})\n"
                "# F2: an adapter the pre-registered rule did not select is never exported under the pre-registered names.\n"
                "adapter_tag = run_tag + ('_not_selected' if globals().get('grid_selection') and grid_selection['keep_frozen'] else '')\n"
                "if adapter_tag != run_tag:\n"
                "    print({{'selected_model': 'frozen', 'note': f'the adapter was not selected (keep-frozen rule); Sections 9 and 10 write it with the {{adapter_tag}} tag'}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'instruction': INSTRUCTION,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'near_duplicates': near_duplicates,\n"
                "    'n_documents': len(document_set),\n"
                "    'baselines': {{'random_floor': floor, 'lexical': baseline_lexical}},\n"
                "    'test_look': look,\n"
                "    'preregistered': preregistered,\n"
                "    'settings_differing': settings_differing,\n"
                "    'run_settings': run_settings,\n"
                "    'device': str(pipe.device),\n"
                "    'dtype': MODEL_DTYPE,\n"
                "    'grid_selection': globals().get('grid_selection'),\n"
                "    'reported_split': report_label,\n"
                "    'frozen_reported_split': frozen_report,\n"
                "    'frozen_validation': frozen_val,\n"
                "    'fresh_test': fresh_result,\n"
                "    'fresh_test_1_exposed': fresh1_result,\n"
                "    'fresh_test_manifest': fresh['manifest'] if fresh else None,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'reported_split_metrics': adapted_report,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "report_path = f'outputs/{stem}_evaluation_report{{run_tag}}.json'\n"
                "with open(report_path, 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': report_path}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the four rows side by side, `delta_vs_frozen`, `median_rank`, and the `verdict`.\n\n<details><summary>Check your '
                'reasoning</summary>Recall@1. On the exposed development test, in the Colab CLI T4 runs of 9 and 17 October 2026 (blobs `26a62b80` and `971187e6`, identical; recipe 1, bfloat16 training on CUDA; not re-run for this revision), recall@1 rose 0.6312 → 0.7662 (+0.135) but recall@10 only 0.9377 → 0.9896 (+0.052): the frozen model '
                'already had the right intent near the top, and fine-tuning mostly moved it to first place. MRR went 0.741 → 0.8498 and the verdict was *improved*. Those queries '
                'had been used to choose the old default, so read the fresh-test row and its interval for this revision; its recorded value is in `docs/release-verification.md` once the hosted run exists.</details>'
            ),
        },
        {
            "md": (
                "## 9. Retrieve before and after, export the adapter and reload it\n\n"
                "The three queries embedded by the frozen model in Section 5 are embedded again by the adapted model "
                "through the same `embed` contract, ranked against the re-embedded 77 documents, and their top-3 lists "
                "printed side by side with the gold intent. Read them as observations: the metric is Section 8, and the "
                "adapter moves the last decoder layers so **every vector changes** — each document's cosine to its frozen "
                "self is printed too. The per-batch `evaluation_report` helper — the inference-stage helper — is written "
                "for the probe queries and stays `not-measurable`, because a batch of vectors has no metric without a "
                "labelled set; `pipe.evaluate` is that labelled evaluation.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the last two decoder layers, about 126 MB in float32 (recipe 1's bfloat16 T4 adapter was 63 MB) — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the instruction it was trained with, the tensor "
                "names, the file size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`Qwen3EmbeddingPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest and "
                "digest **before** deserialising, refuses any tensor that is not a decoder-layer tensor of the base, and "
                "overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). "
                "The cell asserts identical query vectors and an identical validation MRR (VER4). Files of an exploratory run carry its `_exploratory_<n>` tag, so they never replace the pre-registered run's."
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
                "with open(f'outputs/{stem}_retrieval{{adapter_tag}}.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(rows)\n\n"
                "artifact_dir = Path(f'outputs/{stem}_adapter{{adapter_tag}}')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'instruction': artifact_manifest['adapter']['instruction'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = Qwen3EmbeddingPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_queries = np.asarray(reloaded.embed(probe_queries, kind='query', instruction=INSTRUCTION)['embeddings'], dtype=np.float32)\n"
                "# Reload parity on the validation queries (never a test split).\n"
                "reloaded_val = reloaded.evaluate(val_records, instruction=INSTRUCTION, candidates=val_documents)\n"
                "in_memory_val = pipe.evaluate(val_records, instruction=INSTRUCTION, candidates=val_documents)\n"
                "parity = {{'query_vectors_identical': bool(np.array_equal(query_vectors_after, reloaded_queries)), 'mrr_in_memory': round(in_memory_val['mrr'], 6), 'mrr_reloaded': round(reloaded_val['mrr'], 6), 'split': 'validation'}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['query_vectors_identical'] and abs(in_memory_val['mrr'] - reloaded_val['mrr']) < 1e-9\n\n"
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
                "    'test_look': look,\n"
                "    'preregistered': preregistered,\n"
                "    'settings_differing': settings_differing,\n"
                "    'grid_selection': globals().get('grid_selection'),\n"
                "    'fresh_test': fresh_result,\n"
                "    'fresh_test_1_exposed': fresh1_result,\n"
                "    'fresh_test_manifest_sha256': fresh['manifest']['manifest_sha256'] if fresh else None,\n"
                "    'retrieval_before_after': rows,\n"
                "    'document_shift': document_shift,\n"
                "    'batch_report': single_report,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': MODEL_DTYPE}},\n"
                "}}\n"
                "with open(f'outputs/{stem}_result{{run_tag}}.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** the frozen and adapted top-3 for the three probes, `document_shift`, the artifact's size, and `reload_parity`.\n\n<details><summary>Check "
                'your reasoning</summary>Every vector moves, because the adapter changes layers all inputs share: the self-cosine of each document to its frozen '
                'version is below 1, so cosines from the two models are not comparable. In the recorded run the reloaded adapter gave identical query vectors and the '
                'same MRR on the 154 validation queries in the T4 run of 17 October 2026 (blob `971187e6`, recipe 1: 0.787748 in memory and reloaded, vectors identical).</details>'
            ),
        },
        {
            "md": (
                "## 10. Optional transfer check: CLINC150 (display only)\n\nBanking77 may be in the model's training data (DAT9), so this cell asks two questions on a corpus that is **not an MTEB task**, CLINC150 [5] (150 intents in ten domains). `fetch_transfer_corpus` downloads its `data_full.json` at a pinned commit and refuses a size or SHA-256 mismatch; `build_transfer_set` draws 2 in-scope test queries per intent (300) and refuses any draw but the pinned one. Two of the ten domains, `banking` and `credit_cards`, are close to Banking77's own (balance, transfer, card declined, PIN change ...), so the check is reported in two rows, each ranked against its own intent phrases: **near-domain transfer** (60 queries, 30 phrases) and **the other 8 domains** (240 queries, 120 phrases: travel, kitchen, small talk, work ...), which is the only row that says anything about *other domains*. Both models embed with the same instruction (`TRANSFER_INSTRUCTION`); each row prints recall@1 and MRR beside the random floor and the lexical baseline, and the paired MRR difference with a query-level and an intent-level bootstrap 95 % interval. **Display only:** it is never a selection input and never a stop; whether CLINC150 was in Qwen3-Embedding's training data is undisclosed too, so it is a transfer check, not an out-of-distribution guarantee. It took 11 s on a T4 under recipe 1. Set `RUN_TRANSFER_CHECK = False` to skip it.\n\n**Predict before running:** the adapter was trained on 77 banking intents. Will it help more on the near-domain row or on the other 8 domains, and could it hurt there?"
            ),
            "code": (
                "RUN_TRANSFER_CHECK = True  # @param {{type:\"boolean\"}}\n"
                "\n"
                "if not RUN_TRANSFER_CHECK:\n"
                "    print({{'transfer_check': 'not run (RUN_TRANSFER_CHECK is False)'}})\n"
                "elif pipe.adapter is None:\n"
                "    raise RuntimeError('The pipeline holds the pinned base, not an adapted model: run Section 7, Section 8 and Section 9, then this cell.')\n"
                "else:\n"
                "    transfer = build_transfer_set(fetch_transfer_corpus(cache_dir='weights/clinc150'))\n"
                "    t0 = time.perf_counter()\n"
                "    transfer_rows = {{}}\n"
                "    for label, group in transfer['groups'].items():\n"
                "        g_floor = random_floor(len(group['documents']))\n"
                "        g_lexical = pipe.lexical_baseline(group['records'], group['documents'])\n"
                "        g_frozen = pipe.evaluate_base(group['records'], instruction=TRANSFER_INSTRUCTION, candidates=group['documents'])\n"
                "        g_adapted = pipe.evaluate(group['records'], instruction=TRANSFER_INSTRUCTION, candidates=group['documents'])\n"
                "        g_intents = [r['intent'] for r in group['records']]\n"
                "        transfer_rows[label] = {{'n_queries': g_frozen['n_queries'], 'n_documents': g_frozen['n_documents'], 'random_floor': g_floor, 'lexical': g_lexical, 'frozen': g_frozen, 'adapted': g_adapted, 'paired_mrr': paired_rank_comparison(g_frozen['ranks'], g_adapted['ranks']), 'paired_mrr_intent_cluster': paired_cluster_comparison(g_frozen['ranks'], g_adapted['ranks'], g_intents)}}\n"
                "        row = transfer_rows[label]\n"
                "        print({{label: {{'queries': row['n_queries'], 'phrases': row['n_documents'], 'random_floor_mrr': round(g_floor['mrr'], 4), 'lexical': brief(g_lexical), 'frozen': brief(g_frozen), 'adapted': brief(g_adapted)}}}})\n"
                "        print({{label + ': paired MRR': {{'delta': round(row['paired_mrr']['delta_mrr'], 4), 'ci95_queries': [round(x, 4) for x in row['paired_mrr']['ci']], 'ci95_intents': [round(x, 4) for x in row['paired_mrr_intent_cluster']['ci']], 'improved': row['paired_mrr']['n_improved'], 'worse': row['paired_mrr']['n_worse']}}}})\n"
                "    print({{'seconds': round(time.perf_counter() - t0, 1), 'role': 'display only; not a selection input or a stop'}})\n"
                "    transfer_payload = {{\n"
                "        'corpus': TRANSFER_NAME, 'release': TRANSFER_RELEASE, 'url': TRANSFER_URL, 'file': {{'name': TRANSFER_FILE[0], 'bytes': TRANSFER_FILE[1], 'sha256': TRANSFER_FILE[2]}}, 'license': TRANSFER_LICENSE,\n"
                "        'manifest_sha256': transfer['manifest_sha256'], 'instruction': TRANSFER_INSTRUCTION, 'near_domains': list(TRANSFER_NEAR_DOMAINS), 'groups': transfer_rows,\n"
                "        'adapter': {{k: pipe.adapter[k] for k in ('trainable_layers', 'lr', 'best_epoch') if k in pipe.adapter}}, 'adapter_selected': adapter_tag == run_tag, 'role': 'display only; not a selection input or a stop', 'run_test_look': look,\n"
                "    }}\n"
                "    with open(f'outputs/{stem}_transfer_check{{adapter_tag}}.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump(transfer_payload, handle, indent=2, ensure_ascii=False)\n"
                "    print({{'transfer_report': f'outputs/{stem}_transfer_check{{adapter_tag}}.json'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the two rows separately, each MRR difference and whether its intervals exclude 0, and how many queries improved against those that got worse.\n\n<details><summary>Check your reasoning</summary>The recorded T4 run of 17 October 2026 (blob `971187e6`, recipe 1) gave near-domain +0.0227 (95 % CI −0.0097 to 0.0579) and other 8 domains −0.0159 (−0.0348 to 0.0022): both intervals include 0. Recipe 2 has no recorded value yet. Read the rows apart: the near-domain row asks whether the banking adaptation transfers to nearby finance intents; only the other-8-domains row speaks to other domains. There, an interval above 0 means the adaptation also sharpened intent retrieval outside finance, an interval below 0 means it specialised the model at a cost to those domains, and an interval across 0 means this check cannot tell; with 240 queries it can only detect fairly large changes. Pooling the two rows would mix a likely gain with a possible loss. None of it changes the configuration chosen in Section 7.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "Read the fresh-test row of Section 8 first: it is the only number here that played no part in any choice (its recorded hosted value is in "
        "`docs/release-verification.md`), and its intent-level interval is the more cautious of its two intervals. For recipe 1 the record is +0.112 MRR on "
        "fresh test 1 (0.7338 → 0.8455, 95 % CI 0.091 to 0.134; T4, 17 October 2026, blob `971187e6`); this revision's float32 recipe is read on fresh test 2. "
        "In the previous blob's recorded runs the frozen embedder already retrieved the right intent phrase for well over half of the "
        "exposed-development-test messages (recall@1 in the sixties against a lexical baseline in the thirties and a random floor of 1.3 %), "
        "and a bounded contrastive fine-tuning of the last two decoder layers on 616 pairs lifted recall@1 on those queries by more than ten "
        "points and MRR by about a tenth, with a 126 MB float32 adapter (63 MB in recipe 1's bfloat16) that reloads to identical vectors. Because Banking77 may be in the model's "
        "training data (DAT9), the frozen number may be in-distribution, and because those queries chose the old default, the fresh-test row "
        "and its interval are the numbers to read for this revision. That is "
        "the claim: the adaptation contract can adapt the embedder to a retrieval task end to end on a real labelled "
        "corpus, and the numbers it produces are read against a random floor, a lexical baseline and the frozen model "
        "rather than in isolation.\n\n"
        "The fresh test is 770 queries (10 per intent) and the exposed development test 385, against 77 two-word documents from one seeded "
        "split of one corpus; the paired bootstrap interval covers query sampling, not seed-to-seed variation; recall@k and MRR say whether the gold document ranks first, not whether the vectors are good "
        "for any other use. The adapter changes the last layers, which every input shares, so every vector shifts (Section "
        "9 prints each document's cosine to its frozen self) and cosine values from the adapted model are not comparable "
        "to the frozen model's; nothing here measures the effect on other tasks. Since amendment 1 the model runs in float32 on every "
        "device, so a GPU run should be close to the CPU float32 numbers but not identical to the last digit; recipe 1 trained in bfloat16 on CUDA and fell 0.03–0.10 MRR below CPU on validation.\n\n"
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
        "model on a fresh test split scored only under the pre-registered settings, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, embedding quality on any other task, a usable "
        "acceptance threshold, or production fitness.\n\n"
        "## Activity: how much does the second trainable layer buy?\n\n"
        "Run this only after the default Run all has finished; Section 9's exported files stay as they are unless you re-run Section 9.\n\n"
        "1. **Predict.** Write down the held-out recall@1 you expect from the adapted model if only the **last one** decoder layer trains "
        "(half the trainable parameters, a 63 MB adapter). Above or below the default's? By how much?\n"
        "2. **Change.** In the training cell of Section 7 (7b) set `TRAINABLE_LAYERS = 1`; leave every other field as it is. This changes a pre-registered "
        "setting, so the run is **exploratory**: Section 8 reads validation and the exposed development test only and writes `_exploratory_<n>` files.\n"
        "3. **Run.** Run Section 7 (the grid cell need not run again), then Section 8. Every adaptation starts from the pinned base (`started_from` in Section 7's output), "
        "so this is a fresh experiment, not more training on top of the first run.\n"
        "4. **Observe.** Section 7's epoch 0 (`note: frozen model`) must print the same validation MRR as in the default run — the same "
        "starting point. Then read `delta_vs_frozen` in Section 8 and compare it with the default's.\n"
        "5. **Explain.** Say in one sentence what the second layer added, and whether that is worth twice the adapter size for this task.\n\n"
        "To put the notebook back to the pre-registered state, set `TRAINABLE_LAYERS = 0` (the grid's selection) and run Sections 7b, 8 and 9 again.\n\n"
        "<details><summary>Check your reasoning</summary>Expect a smaller gain, not none: the frozen model already ranks the right intent near the top, "
        "and one layer can still reorder the last few places, so recall@1 still rises but by less than the old default's +13.5 points (exposed development test, previous blob's T4 runs of 19 September and 9 October 2026). "
        "The exact figure depends on the device; what matters is the direction and the adapter-size trade-off.</details>\n\n"
        "**More experiments (same procedure, they do not affect the default path):** set `TEMPERATURE = 0.2` and read whether the softer "
        "contrast learns as fast; set `EPOCHS = 4` and watch whether validation MRR keeps rising or turns (the best epoch is kept either way); "
        "change `INSTRUCTION` in Section 4, **run Section 4**, then Sections 5–6 to re-read the frozen numbers (to adapt as well, set `RUN_GRID = False` first: the run is exploratory anyway) — the model is instruction-aware; or bring "
        "your own pairs through BYOD (re-run from Section 4) and read the lexical baseline before the adapted number. Each of these is "
        "exploratory: Section 8 reports validation and the exposed development test, never the fresh test, and the files carry an `_exploratory_<n>` tag.\n\n"
        '## Troubleshooting\n\n- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google '
        'Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 '
        'again; a complete environment built from the same lock is reused, an incomplete one is finished. If it repeats, the network is blocking or altering '
        '`files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and '
        'choose **Run all**.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it '
        'keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message '
        'names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again.\n- **Section 4 reports a size or SHA-256 mismatch for a '
        'Banking77 file** — the download from `raw.githubusercontent.com` was cut short or altered. Run Section 4 again; a verified cached file is reused.\n- '
        '**Section 7 is slow** — on CPU each grid configuration takes several minutes and the grid trains three; switch to a T4 GPU for a faster run (float32 there too since amendment 1).\n- **Your numbers differ from the recorded run** — the recorded hosted run used recipe 1 (bfloat16 training on CUDA); this revision '
        'trains in float32, so adapted numbers will differ by more than rounding, and CPU and GPU float32 still differ in low-order digits; the comparison between systems, '
        'not the last digit, is the result.\n- **BYOD: \"CSV is missing columns\" or a record refusal** — the message names the rule: columns `id`, `query`, '
        '`positive` (optional `negative`), at least 12 distinct queries (8 stay for training) and two distinct positives; set `INSTRUCTION` to describe your retrieval task.\n- **BYOD: '
        '"BYOD_PATH … is not a file"** — the path is relative to the working directory printed in the message; give one .csv, .json or .jsonl file.\n- **BYOD: '
        '"the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the file in the runtime and set `BYOD_PATH` to its path.\n- **BYOD: '
        '"Upload exactly one file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **Section 8 or 9 says the pipeline '
        'holds the pinned base** — you re-ran Section 5 or 6 after Section 7, and they put the pretrained weights back so their numbers are the '
        'frozen model\'s. Run Section 7 again, then Section 8 and Section 9.\n- **Section 7b or 8 says the run is exploratory** — a setting differs from the pre-registration (named in '
        '`settings_differing`); that is expected for the Activity, the experiments and BYOD. Set the Section 7b fields back (`LEARNING_RATE = 0`, `TRAINABLE_LAYERS = 0`, '
        'the other defaults), make sure the Section 7a grid ran in this session, and run Sections 7b, 8 and 9 again for a pre-registered pass.\n- **Out of memory in Section 7** — '
        'choose a GPU runtime; a smaller `BATCH_SIZE` works too but makes the run exploratory.\n- **BYOD refused** — the message names the file and the rule; fix the file '
        'and run Section 4 again. Files already in `outputs/` come from an earlier run.\n\n## Glossary\n\n- **Embedding:** a fixed-length '
        'vector (1024 numbers here) that places a text so similar meanings are close; it carries no score by itself.\n- **Last-token pooling / L2 '
        'normalisation:** the vector is the hidden state of the final token, scaled to length 1 so cosine equals the dot product.\n- **Instruction-aware '
        'query:** queries are prefixed with a one-line task description; documents are embedded as-is.\n- **Cosine similarity:** the angle between two unit '
        'vectors; a similarity, not a probability, and no threshold ships.\n- **Recall@k:** the share of queries whose correct document is among the top k; '
        '**MRR** is the mean of 1 / rank of the correct document.\n- **Random floor / lexical baseline:** what a random ranking achieves; ranking by shared-word '
        '(Jaccard) overlap — what no model at all gets.\n- **Exposed development test / fresh test:** the 385 test queries the old default was chosen on, shown '
        'for continuity; the 770 never-scored test queries read only under the pre-registered settings.\n- **Pre-registered / exploratory:** a run '
        'whose every setting matches the written-down plan; any other run, compared on validation only.\n- **InfoNCE with in-batch negatives:** a contrastive loss that asks each query to pick its own positive '
        "among the batch's documents; the other queries' positives are the negatives.\n- **Temperature:** the scale applied to cosines in the loss; lower is a "
        'sharper contrast.\n- **Frozen / adapted / pinned base:** the packaged model; the model after Section 7; the verified packaged weights every adaptation '
        'starts from (`restore_base`).\n- **Query-disjoint split:** no message appears in two splits, ignoring case, punctuation and spacing; near-paraphrases (token Jaccard ≥ 0.8) are counted, not removed.\n- **Isolated environment:** the '
        'separate Python environment Section 1 builds from the hash lock; every later cell runs there.\n\n## Conclusion (your notes)\n\nComplete these in your own '
        "words; the labelled recorded values are in the **Check your reasoning** answers above.\n\n- On the fresh test (Section 8) the frozen MRR was ___ and the selected model's ___; the paired difference was ___ with a 95 % interval of "
        "___ (queries) and ___ (intents) — or: not scored, because my run was exploratory (`settings_differing`: ___).\n- Secondary, on the 385 "
        "exposed-development-test queries that chose the old default: recall@1 went from ___ (frozen) to ___ (adapted) against the lexical baseline's ___ "
        '(verdict: ___).\n- The number I would not trust on its own is ___, because ___.\n- Before adapting on my own pairs I would de-duplicate by ___, compare '
        'against ___, and use harder negatives such as ___.\n\n'
        "## References\n\n"
        "[1] Y. Zhang et al., \"Qwen3 Embedding: Advancing text embedding and reranking through foundation models,\" arXiv:2506.05176, 2025. [Online]. Available: https://arxiv.org/abs/2506.05176\n\n"
        "[2] Qwen Team, \"Qwen3-Embedding,\" GitHub repository, evaluation/task_prompts.json at commit 44548aa5f0a0aed1c76d64e19afe47727a325b8f. [Online]. Available: https://github.com/QwenLM/Qwen3-Embedding\n\n"
        "[3] Qwen Team, \"Qwen/Qwen3-Embedding-0.6B,\" Hugging Face model repository, revision {MODEL_REVISION}. [Online]. Available: https://huggingface.co/{MODEL_ID}\n\n"
        "[4] I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, and I. Vulić, \"Efficient intent detection with dual sentence encoders,\" in *Proc. 2nd Workshop on Natural Language Processing for Conversational AI*, 2020, pp. 38–45. [Online]. Available: https://arxiv.org/abs/2003.04807 (Banking77, CC BY 4.0)\n\n"
        "[5] S. Larson et al., \"An evaluation dataset for intent classification and out-of-scope prediction,\" in *Proc. EMNLP-IJCNLP*, 2019, pp. 1311–1316. [Online]. Available: https://aclanthology.org/D19-1131 (CLINC150, CC BY 3.0; data at https://github.com/clinc/oos-eval)\n\n"
        "[6] N. Muennighoff, N. Tazi, L. Magne, and N. Reimers, \"MTEB: Massive text embedding benchmark,\" in *Proc. 17th Conf. European Chapter of the Association for Computational Linguistics (EACL)*, 2023, pp. 2014–2037. [Online]. Available: https://github.com/embeddings-benchmark/mteb\n\n"
        "[7] K. Valcorza, \"qwen3-embedding-pipeline: README, model card, weight provenance, pre-registration and release verification,\" GitHub repository. [Online]. Available: https://github.com/kurtvalcorza/qwen3-embedding-pipeline (README.md, MODEL_CARD.md, docs/WEIGHTS.md, docs/qwen3-embedding-preregistration.md, docs/release-verification.md)\n\n"
        "[8] DIMER platform, \"DIMER Notebook Specification 2.5\" and \"Model Card Specification 1.1,\" fleet specifications in the ml-worker repository, 2026."
    ),
}
