# Release verification

`tutorials/qwen3_embedding_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `samples.py`, `metrics.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 11-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned Banking77
  commit is the one allowed second hash);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Qwen3EmbeddingPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `check_split_disjoint`, `write_dataset_csv`, `documents`, `validate_inputs` with the oversized-batch refusal probe, `pipe.embed` on queries and documents
  with the sanity checks plus the frozen top-3 lists, `random_floor`, `pipe.lexical_baseline`, `pipe.evaluate` on
  the frozen model with the floor assertion and on the validation and test splits after adaptation with the MRR
  assertion, `pipe.adapt` with its explicit hyperparameters, `trainable_layers=TRAINABLE_LAYERS`,
  `temperature=TEMPERATURE` and `instruction=INSTRUCTION`, the per-batch `evaluation_report`,
  `pipe.save_artifact`, `Qwen3EmbeddingPipeline.from_artifact` and the reload-parity assertion, and the provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the six expected
  `outputs/` paths, the learner-facing statements (representations not predictions, adaptation measured by retrieval, the
  random floor, the lexical baseline, cosine is a similarity not a probability, InfoNCE, no dispersion estimate,
  every vector changes, named exclusions, the CC BY 4.0 corpus licence) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `from transformers import` / `AutoModel` / `AutoTokenizer` /
  `last_hidden_state` / `from huggingface_hub import` / `urllib.request` / `safetensors` / `torch.optim` /
  `.backward(` / `pipe._model` / `cross_entropy(` use **outside the carried module cells**, `trust_remote_code=True`,
  `pickle.load`, `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI installs only `pytest`, `ruff` and `numpy` plus the package without its model dependencies (no torch, no
transformers), runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected runner, tokenizer, scorer and corpus fetcher, temporary manifests, no weights
— `tests/test_model_backed.py` is skipped without `transformers` and the snapshot). These are source/provenance and unit
checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; bfloat16 there, float32 on CPU) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/qwen3-embedding-0.6b/` or the corpus cache `weights/banking77/` (the standalone path writes the
   manifest itself, stages the missing file from the Hub, and fetches the two pinned Banking77 files from the project
   repository, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `INSTRUCTION = 'Given a customer support message, retrieve the banking intent
   it expresses'`, `EPOCHS = 2`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 16`, `TRAINABLE_LAYERS = 2`,
   `TEMPERATURE = 0.05`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3` (an interpreter restart after the install is expected where the runtime's preinstalled torch or
   numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `Qwen3EmbeddingPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `cosine_similarity`, `fetch_corpus`,
     `read_corpus`, `build_sample_dataset`, `filter_records`, `validate_dataset`, `documents`,
     `check_split_disjoint`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `retrieval_metrics`,
     `random_floor`, `lexical_baseline` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `Qwen/Qwen3-Embedding-0.6B` at the immutable revision, and `verify_snapshot` returning its dict (11 files);
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory;
   - Section 4: `fetch_corpus` fetching the two pinned files (839,073 / 239,961 bytes) from `raw.githubusercontent.com`
     into `weights/banking77/`, 10,003 + 3,080 raw rows read, and the seeded balanced draw of 616 / 154 / 385 pairs
     over 77 intents with `check_split_disjoint` reporting no shared message, 77 documents, and the three dataset
     digests `dacba395…` / `3d07cfd5…` / `df00c83e…`; `outputs/…_train.csv` written; the four dataset refusal
     probes each raising `ValueError`;
   - Section 5: the ceilings (`MAX_BATCH` 64, `MAX_TEXT_TOKENS` 8192, `MAX_TEXT_CHARS` 100000, `EMBEDDING_DIM` 1024,
     `MAX_TRAIN_TOKENS` 64) surfaced; `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, the
     query manifest with the instruction, one recorded rejection finding from the oversized-batch probe); `embed` on
     the 77 documents and three test queries with all five sanity checks `True` and the frozen top-3 lists printed;
   - Section 6: the random floor (recall@1 1.3 %, MRR ≈ 0.064), the lexical baseline (≈ 30.6 % recall@1 on the
     sample) and the frozen embedder's test metrics (≈ 63.4 % recall@1, MRR ≈ 0.742) on CPU float32,
     with the cell's assertion that the document counts agree and the frozen MRR is above the floor;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 31,461,888 trainable of 595,776,512 parameters,
     77 training documents, and a two-epoch history with validation MRR rising (0.699 → 0.812 → 0.828 in the recorded run;
     `best_epoch` 2);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison and
     `outputs/…_evaluation_report.json` written (the cell asserts the adapted test MRR exceeds the frozen one — on the
     sample recall@1 ≈ 80.3 % versus ≈ 63.4 %);
   - Section 9: the three test queries retrieved again by the adapted embedder and printed beside the frozen top-3 and
     the gold intent, each document's cosine to its frozen self summarised, the per-batch `evaluation_report` verdict
     `not-measurable`, `outputs/…_retrieval.csv` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors, manifest.json}` (22 tensors, about 126 MB, `instruction`
     recorded) and `Qwen3EmbeddingPipeline.from_artifact` reloading it with identical query vectors and an identical
     77-query MRR (the cell asserts both); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity
     and licence, the snapshot block (`weight_format`, `weight_sha256`), the instruction, the `corpus` block, the
     inference-contract items, the comparison, the before/after retrievals, the document shift, the artifact digest,
     the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable
   `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `qwen3_embedding_colab.ipynb` (`E2E`) | `9bcb449` / `40e0a136` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `qwen3_embedding_colab.ipynb` (`TASK-INFERENCE`, superseded) | `cbeec85` / `7715612eea32` | 2026-09-14 | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-embedding` v2) | PASSED — 8/8 code cells, 234.2 s, 1,207 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/qwen3_embedding_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/qwen3_embedding_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include the model download where it occurred; they are measurements for the stated runtime, not
general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `9bcb449` / `40e0a136` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 11 entries because the snapshot was pre-staged → `verify_snapshot` 11 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest checks → 10,003 + 3,080 rows read, 616 / 154 / 385 drawn over 77 intents with `check_split_disjoint` clean, 77 documents and digests `dacba395…` / `3d07cfd5…` / `df00c83e…` → four dataset refusals → input manifest with the oversized-batch refusal → `embed` of the 77 documents (0.72 s) and three test queries (0.18 s; 30 / 48 / 36 tokens) with all five sanity checks `True` → frozen top-3 lists → random floor → lexical baseline → frozen evaluation → `adapt` → validation + test evaluation → retrievals after adaptation → adapter export → reload parity) | 226.9 s | **PASSED** — 11/11 code cells; random floor recall@1 1.3 % / MRR 0.064; lexical baseline recall@1 30.6 %, recall@5 50.4 %, MRR 0.416, median rank 5; frozen test recall@1 63.4 %, recall@5 88.6 %, recall@10 94.0 %, MRR 0.742, median rank 1 (30.7 s); `adapt` 31,461,888 of 595,776,512 params, 616 pairs over 77 documents, 2 epochs, 133.3 s, validation MRR 0.699 → 0.812 → 0.828 (recall@1 58.4 → 70.1 → 72.7 %; `best_epoch` 2, train loss 0.479 → 0.178); **adapted test recall@1 80.3 %, recall@5 97.9 %, recall@10 99.5 %, MRR 0.879 (Δ +16.9 / +9.4 / +5.5 points, +0.136 MRR)**; the three probe queries ranked their gold intent first before and after (the top-3 below the gold changed for all three); document self-cosine to the frozen vectors median 0.680, minimum 0.520; per-batch report `not-measurable`; adapter 125,849,896 B / 22 tensors, SHA-256 `ac38839e…`; reload parity exact (query vectors identical, 77-query MRR 0.859199 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `cbeec85` / `7715612eea32` (`TASK-INFERENCE`, superseded) | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-embedding` v2) | Default sample path of the inference-only notebook: the four README sentences, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 11 files, query/document embedding with the cosine check, `not-measurable` report, CSV + JSON exports | 234.2 s | **PASSED** — 8/8 code cells, 1,207 MB staged; history only |

## Current status

The `E2E` notebook source is complete and passes all static checks, including the generator parity checks
(`--check` OK). A local pre-flight execution of the committed blob completed the whole default path on CPU — corpus
read from the cache, validation and split, the embedding contract, the random floor, the lexical baseline and the
frozen retrieval metrics, two epochs of contrastive fine-tuning, held-out evaluation, before/after retrievals, adapter
export and reload parity — which catches defects but is **not** a supported runtime under REL1/REL10, and it ran with
the snapshot and the two Banking77 files pre-staged, so neither the 1.19 GB Hub fetch nor the corpus download has been
exercised by this notebook end to end; the earlier `TASK-INFERENCE` Kaggle run did exercise the Hub fetch and digest
check of the same snapshot. The repository stays at **Candidate** until a Colab or fresh-container run of the exact
`E2E` release revision is recorded above.
