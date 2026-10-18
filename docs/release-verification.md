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

CI also installs the pinned CPU-only torch wheel plus `transformers`, `huggingface-hub`, `safetensors` and `numpy`,
runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`, `tests/test_import_boundary.py`,
`tests/test_notebook_parity.py`; injected bag-of-words runner and corpus fetcher, temporary manifests, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

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
| `qwen3_embedding_colab.ipynb` (`E2E`) | `9ca75dc` / `de48070a` | 2026-10-18 | Colab CLI 0.7.4, fresh Colab Tesla T4 (sequential `colab exec -f`, not a browser Run all), `cuda:0`, float32 | **PASSED** — 14/14 code cells, 0 errors, no restart, 346.8 s; **REL13: pre-registered evaluation of record for recipe 2** on fresh test 2 (record section *2026-10-18 hosted T4 evaluation of record, recipe 2* below); the 2026-10-17 row stays the recipe-1 record |
| `qwen3_embedding_colab.ipynb` (`E2E`) | `1cc43ed` / `971187e6` | 2026-10-17 | Colab CLI 0.7.4, fresh Colab Tesla T4 (sequential `colab exec -f`, not a browser Run all), `cuda:0`, bfloat16 | **PASSED** — 14/14 code cells, 0 errors, no restart, 382.9 s; **REL13: pre-registered evaluation of record** for the fresh test (record section *2026-10-17 hosted T4 evaluation of record* below) |
| `qwen3_embedding_colab.ipynb` (`E2E`) | `ec6dd95` / `f3475b93` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-qwen3-embedding` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 26 files, 1209 MB staged from the Hub into a clean cache; comparison {recall@1: {random_floor: 0.013, lexical: 0.3065, frozen: 0.6312, adapted: 0.7662}, recall@5: {random_floor: 0.0649, lexical: 0.5039, frozen: 0.8857, adapted: 0.9714}, recall@10: {random_floor: 0.1299, lexical: 0.6753, frozen: 0.9377, adapted: 0.9896}, mrr: {random_floor: 0.064, lexical: 0.4159, frozen: 0.741, adapted: 0.8498}, median_rank: {lexical: 5, frozen: 1, adapted: 1}, delta_vs_frozen: {recall@1: 0.1351, recall@5: 0.0857, recall@10: 0.0519, mrr: 0.1087}}; reload parity {query_vectors_identical: True, mrr_in_memory: 0.8125, mrr_reloaded: 0.8125}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-qwen3-embedding/v3/evidence/` in the workspace |
| `qwen3_embedding_colab.ipynb` (`E2E`) | `9bcb449` / `40e0a136` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `qwen3_embedding_colab.ipynb` (`TASK-INFERENCE`, superseded) | `cbeec85` / `7715612eea32` | 2026-09-14 | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-embedding` v2) | PASSED — 8/8 code cells, 234.2 s, 1,207 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/qwen3_embedding_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/qwen3_embedding_colab.ipynb`). Wall times are the sum of per-cell times reported by
the executor and include the model download where it occurred; they are measurements for the stated runtime, not
general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-18 (00:38:29 UTC start) | commit `9ca75dc` / blob `de48070a3621ed61529948a1b1f3a0e671bf9ed4` (PR #14 head; `NOTEBOOK_SOURCE.repository_revision` `f7f3dc0`; generator `build_notebook.py/2.2`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab VM (session `suite-qwen3-9ca75dc-3e4c`), Tesla T4; blob fetched at the 40-char SHA and Git-blob verified, no repository checkout; kernel Python 3.13.15; uv isolated environment CPython 3.12.12 (47 locked packages, 51 s); `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, float32 | Default sample path only (`USE_BYOD = False`, default form values, `RUN_GRID` on); 11 snapshot files fetched and digest-verified | 346.8 s | **PASSED — REL13 pre-registered evaluation of record, recipe 2** (`preregistered` True, `settings_differing` []): 14/14 code cells, 0 errors; grid C0 / C1 / C2 validation MRR 0.8284 / 0.7962 / 0.827 → C0; fresh test 2 (770, `cd017128…`) frozen 0.613 / 0.7274, adapted 0.7727 / 0.8595 (recall@1 / MRR), Δ MRR +0.1321 [0.111, 0.1547] queries, [0.0886, 0.1748] intents; fresh test 1 (exposed) 0.7324 → 0.8666; exposed development test 0.7422 → 0.8785; reload parity exact (0.828415); adapter 125,849,896 B. Evidence [`docs/execution-evidence/2026-10-18-9ca75dc/`](execution-evidence/2026-10-18-9ca75dc/): executed notebook SHA-256 `5e4be566f847e0abf6e1646ea5197d78b3199b7ef21fc26670ed2b3bbe3a1843`, `exec.log` `5040de3b18b8e60eec8a59ce3cd1489e51aabbb55bf63899c19fde561e327bd5`, `run_summary.json` `b906a6440e7784aca9461147ed6e1106dfc8e8dd2a91f6ca1bdf1faeb81dc419`. Not exercised: BYOD (REL12), the activity and the optional experiments, a browser Run all |
| 2026-10-17 (17:42:18 UTC start) | commit `1cc43ed` / blob `971187e6a67b878a6ce11c19c0b297c9d512d719` (PR #14 head; `NOTEBOOK_SOURCE.repository_revision` `6a90df9`; `6a90df9..1cc43ed` are the regenerated notebook and a docs-only commit; generator `build_notebook.py/2.2`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab VM, Tesla T4; kernel Python 3.13.15; uv isolated environment CPython 3.12.12 (47 locked packages, built in 44 s), `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, bfloat16 | Default path only, form values at their defaults (no `# @param` edits; the fetched source is byte-equal to the blob); 11 snapshot files fetched from the Hub and digest-verified; `preregistered` True, `settings_differing` [] | 382.9 s (session wall, `run_summary.json`) | **PASSED — REL13: pre-registered evaluation of record** (first hosted run of the pre-registered settings; first scoring of the fresh test by any run): 14/14 code cells, 0 error outputs, no restart; grid C0 / C1 / C2 validation MRR 0.7877 / 0.7197 / 0.7299 (frozen 0.6962) → C0; **fresh test (770): frozen recall@1 0.6351 / MRR 0.7338 → adapted 0.761 / 0.8455, paired MRR difference +0.1118 (95 % CI 0.0911 to 0.134 query-level, 0.0711 to 0.1569 intent-level; 215 improved, 39 worse)**; exposed development test 0.6312 / 0.741 → 0.7662 / 0.8498; CLINC150 near-domain +0.0227 (CI −0.0097 to 0.0579), other 8 domains −0.0159 (CI −0.0348 to 0.0022). Evidence [`docs/execution-evidence/2026-10-17-1cc43ed/`](execution-evidence/2026-10-17-1cc43ed/): executed notebook SHA-256 `fa6a801d4dd305b69b0aa36be805316e97e09769c4ff5d80f8a9314770e3bbdf`, `run_summary.json` `e411bcee379b7a40f9803ea65252e71f8beeb3b4d7f06bbc9a69ed30af2b3f2c`, `exec.log` `fe1ba2865426298a65a4973d1ff984ced0764a009517c88d95bf9497a8a023c0`. Not exercised: BYOD (REL12), the activity and the optional experiments, a browser Run all |
| 2026-10-09 (02:00:47 UTC start) | commit `6606b82` / blob `26a62b8096aa26ad870987d1fa5c0c17b7cf5753` (`NOTEBOOK_SOURCE.repository_revision` `9c8dc93`, the source revision the notebook was generated from; `9c8dc93..6606b82` are the review-fix commit and a test-only commit; generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab VM, Tesla T4; kernel Python 3.13.15; uv isolated environment CPython 3.12.12 (47 locked packages, built in 55 s), `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, bfloat16 | Default sample path only (USE_BYOD = False, default form values); 11 snapshot files fetched from the Hub and digest-verified | 201.4 s | **REL13: earlier-blob history** (blob `26a62b80`, before the pre-registration). **PASSED — one pass, no restart, 0 errors**: 12/12 code cells (the three carried-module cells print nothing); near-duplicates validation 5 / test 9 (Jaccard ≥ 0.8); input manifest 77 documents; random floor recall@1 0.013 / MRR 0.064; lexical 0.3065 / 0.4159; frozen 0.6312 / 0.741 (`adapted: False`); validation MRR 0.6962 → 0.7664 → 0.7877, best epoch 2; adapted 0.7662 / 0.8498 (Δ recall@1 +0.1351, MRR +0.1087, verdict *improved*); document self-cosine median 0.8458; reload parity exact (query vectors identical, MRR 0.812533 both ways); six exports. These match the worked answers. Evidence [`docs/execution-evidence/2026-10-09-6606b82/`](execution-evidence/2026-10-09-6606b82/): executed notebook SHA-256 `f3cc25d398b92b95083a8f3ac0302ea7057f7a151e62aba55a6231bad2a98a8b`, `run_summary.json` `d4c7af9a0a6b05f63000b94af356c9ebd90960c1efd375985a7cd5c13a31fbfb`, `exec.log` `fca046a13c129c2939c670a9c5084dc75e638ce1135a20bbb224668f8769e79d`. Not exercised: BYOD (REL12), the activity and the optional experiments, a browser Run all |
| 2026-10-14 (23:27:48 UTC start) | commit `7dbcc38` / blob `089696de` (`DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb`, PR #13 head; `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab VM, Tesla T4; kernel Python 3.13.15; uv isolated environment CPython 3.12.12 (47 locked packages, built in 52 s), `torch 2.14.0+cu130`, `transformers 4.57.6`, embedder and reranker both on `cuda:0`, bfloat16 | Default path only (USE_BYOD=False, RUN_K_SWEEP=False, seed 42, K=6; no toggles changed); both snapshots fetched from the Hub and digest-verified | 164.5 s (session wall, `run_summary.json`) | **PASSED** — 19/19 code cells in order, 0 error outputs, no restart; embedding recall@1 0.6104 / MRR over 77 0.7310; two-stage recall@1 0.6234 / MRR@6 0.7379; helped / hurt 29 / 24; every computed number equal to the 2026-10-04 run of `d72ea820`; new paired view +17 / −15 (exact sign test p = 0.86), embedding top-6 MRR@6 0.7241; [executed notebook](execution-evidence/2026-10-14/DIMER_Qwen3_Semantic_Search_Reranking_Workshop_7dbcc38_colab-cli-t4.ipynb) |
| 2026-10-04 | `2996bb2` / `d72ea820` (`DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb`) | Google Colab (browser, maintainer-run), Tesla T4; kernel Python 3.13.15, uv isolated environment CPython 3.12.12 (47 locked packages), `torch 2.14.0+cu130`, `transformers 4.57.6`, `cuda:0`, bfloat16 | Default path only (USE_BYOD=False, RUN_K_SWEEP=False, seed 42, K=6; no toggles changed), Run all | not recorded (no per-cell timing saved; isolated setup 65 s, index 11.3 s, rerank 37.2 s) | **PASSED** — 19/19 code cells, execution counts 1..19, 0 error outputs, no restart; two-stage recall@1 0.6234, MRR 0.7379 (identical to the 2026-09-26 run of `5512be30`); [executed notebook](execution-evidence/2026-10-04/DIMER_Qwen3_Semantic_Search_Reranking_Workshop_2996bb2_colab-browser-t4.ipynb) |
| 2026-09-19 | `ec6dd95` / `f3475b93` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-qwen3-embedding` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 323.6 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 26 files, 1209 MB staged from the Hub into a clean cache; comparison {recall@1: {random_floor: 0.013, lexical: 0.3065, frozen: 0.6312, adapted: 0.7662}, recall@5: {random_floor: 0.0649, lexical: 0.5039, frozen: 0.8857, adapted: 0.9714}, recall@10: {random_floor: 0.1299, lexical: 0.6753, frozen: 0.9377, adapted: 0.9896}, mrr: {random_floor: 0.064, lexical: 0.4159, frozen: 0.741, adapted: 0.8498}, median_rank: {lexical: 5, frozen: 1, adapted: 1}, delta_vs_frozen: {recall@1: 0.1351, recall@5: 0.0857, recall@10: 0.0519, mrr: 0.1087}}; reload parity {query_vectors_identical: True, mrr_in_memory: 0.8125, mrr_reloaded: 0.8125}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-qwen3-embedding/v3/evidence/` in the workspace |
| 2026-09-19 | `9bcb449` / `40e0a136` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 11 entries because the snapshot was pre-staged → `verify_snapshot` 11 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest checks → 10,003 + 3,080 rows read, 616 / 154 / 385 drawn over 77 intents with `check_split_disjoint` clean, 77 documents and digests `dacba395…` / `3d07cfd5…` / `df00c83e…` → four dataset refusals → input manifest with the oversized-batch refusal → `embed` of the 77 documents (0.72 s) and three test queries (0.18 s; 30 / 48 / 36 tokens) with all five sanity checks `True` → frozen top-3 lists → random floor → lexical baseline → frozen evaluation → `adapt` → validation + test evaluation → retrievals after adaptation → adapter export → reload parity) | 226.9 s | **PASSED** — 11/11 code cells; random floor recall@1 1.3 % / MRR 0.064; lexical baseline recall@1 30.6 %, recall@5 50.4 %, MRR 0.416, median rank 5; frozen test recall@1 63.4 %, recall@5 88.6 %, recall@10 94.0 %, MRR 0.742, median rank 1 (30.7 s); `adapt` 31,461,888 of 595,776,512 params, 616 pairs over 77 documents, 2 epochs, 133.3 s, validation MRR 0.699 → 0.812 → 0.828 (recall@1 58.4 → 70.1 → 72.7 %; `best_epoch` 2, train loss 0.479 → 0.178); **adapted test recall@1 80.3 %, recall@5 97.9 %, recall@10 99.5 %, MRR 0.879 (Δ +16.9 / +9.4 / +5.5 points, +0.136 MRR)**; the three probe queries ranked their gold intent first before and after (the top-3 below the gold changed for all three); document self-cosine to the frozen vectors median 0.680, minimum 0.520; per-batch report `not-measurable`; adapter 125,849,896 B / 22 tensors, SHA-256 `ac38839e…`; reload parity exact (query vectors identical, 77-query MRR 0.859199 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `cbeec85` / `7715612eea32` (`TASK-INFERENCE`, superseded) | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-embedding` v2) | Default sample path of the inference-only notebook: the four README sentences, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 11 files, query/document embedding with the cosine check, `not-measurable` report, CSV + JSON exports | 234.2 s | **PASSED** — 8/8 code cells, 1,207 MB staged; history only |

## Primary embedding tutorial status

**Current (2026-10-18), recipe 2.** Blob `de48070a` (PR #14 head `9ca75dc`) ran the first hosted pre-registered pass of recipe 2 (float32) on a fresh Colab Tesla T4 (Colab CLI, 14/14 cells, 0 errors, no restart, 346.8 s); that run is the **recipe-2 evaluation of record** on fresh test 2 (MRR 0.7274 → 0.8595, paired +0.1321, 95 % CI 0.111 to 0.1547; record section below). The recipe-1 record that follows is unchanged. Hosted BYOD (REL12) and the activity are not run, so status stays **Candidate** (REL14).

**Recipe 1 (2026-10-17).** Blob `971187e6` (PR #14 head `1cc43ed`) ran its first hosted pre-registered pass on a fresh Colab Tesla T4 (Colab CLI, 14/14 cells, 0 errors, no restart); that run is the **evaluation of record** for the fresh test (record section below). Later hosted runs with identical pre-registered settings are reproductions. Hosted BYOD (REL12) and the activity are not run, so status stays **Candidate** (REL14). The paragraphs below are **REL13: earlier-blob history**.

**Candidate.** The `E2E` notebook blob `f3475b93` (committed at `ec6dd95`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok, 323.6 s, 26 files, 1209 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout, but it needed **one manual restart after the install cell** (pass 1 raised the stale-import guard: cuda-bindings 12.9.4 → 13.4.2, NumPy 2.0.2 → 2.5.3). A restart-assisted run does not meet the no-restart gate (RUN10/ENV6), so that record is history, not release evidence; the earlier "Release-grade" label is withdrawn.

### 2026-10-09 review fixes (QEM-M1..M4, QEM-m1..m4)

The Notebook Review Framework v1 review of `f9f5c81` (`docs/reviews/`) found four Major and four Minor findings. The notebook was regenerated by `tools/build_notebook.py` (/2.2) from `tools/notebook_template.py`:

- **QEM-M1:** Section 1 installs nothing into the kernel. It verifies a pinned `uv` 0.12.15 wheel, builds a managed CPython 3.12.12 environment from `tutorials/requirements-colab.lock.txt` (47 hash-locked packages, `--require-hashes --only-binary :all:`) and routes every later cell to a worker there; the `google.colab` stubs in that worker carry a `ModuleSpec` (`tests/test_worker_colab_stubs.py`). Linux x86_64 only.
- **QEM-M2:** `adapt()` and `load_artifact()` start from the pinned base (`restore_base`); Sections 5 and 6 put the base back before they measure; Sections 8 and 9 refuse to score or export the base as adapted.
- **QEM-M3:** validation and test splits are validated with a one-record floor; `split_dataset` names the dataset size and the true minimum (12 distinct queries) and keeps two documents in each evaluated split; on BYOD the quality comparisons are verdicts, while the Banking77 sample at the default settings still stops on a failed comparison.
- **QEM-M4:** guided layer (audience, how to use, roadmap, predictions, checkpoints, Infrastructure labels, troubleshooting, glossary, conclusion template) and one Predict → Change → Run → Observe → Explain activity.
- **QEM-m1..m4:** `BYOD_PATH`; leakage check ignores case, punctuation and spacing and near-paraphrases are counted; the input manifest covers all 77 documents; Section 7's figures name their device.

Local checks (pytest, Ruff, the validator, both generator `--check`s, and a CPU real-weights run of the regenerated cells with the install skipped) are not clean-runtime evidence. On 2026-10-09 the review-fix blob `26a62b80` (commit `6606b82`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI sequential execution, 12/12 code cells, 201.4 s; held-out recall@1 0.6312 → 0.7662, MRR 0.741 → 0.8498; reload parity exact) — row at the top of *Recorded executions*. That closes the no-restart gate (RUN10/ENV6) for the default path on Colab T4. BYOD on a hosted runtime (REL12) and the activity were not run, so status stays **Candidate** (REL14).

### 2026-10-17 row-72 review fixes (R72-M1..M3, R72-m1..m3)

Pre-registration committed alone first: `docs/qwen3-embedding-preregistration.md` (validation grid C0/C1/C2, margin 0.02,
keep-frozen rule, the pinned 770-query fresh test, score-once wiring, the CLINC150 transfer check). The 385 seed-42 test
queries are the **exposed development test** from this revision on; every number quoted above for them was measured after
the old default had been chosen on them.

**Fresh test.** test.csv after filtering 3,079 → minus 516 exposed (tutorial 385 + workshop W16 154, 23 shared) → 2,563
never scored → minus 22 exact repeats of a train.csv message under `query_key` (the review's normalisation found 7; all 7
are among the 22) → 10 per intent, `random.Random(20261017)` → 770, manifest SHA-256
`fe68b43ae56b9092ab92f6eeb610b2179222d1dfa04c3eebd9c658ac69c0908f`. Near-paraphrases (token Jaccard ≥ 0.8 vs the training/validation draw) are
reported, not removed: 39 in the clean pool, 10 in the draw.

**Development looks at the fresh test: none.** Following SCORE_ONCE_RULE item 8, the fixer's CPU check below ran with
the fresh test masked (built and digest-checked, never scored; a guard aborted the run if any fresh query reached
`evaluate`). The first hosted T4 run of the pre-registered settings is the evaluation of record; later identical hosted
runs are reproductions. `samples.RECORDED_EVALUATION` stays `None` until that run is recorded here. *(2026-10-17: the run is recorded below; the constant is still `None` at `1cc43ed` and is a pending code follow-up, so a later identical run still prints "pre-registered evaluation (this runtime)" in-notebook. This file names the record.)*

**REL13: CPU development look (validation + exposed development test only; fresh test masked, never scored).** Recorded as the CPU build record; never used to change a setting after the hosted run. Windows workstation, CPython 3.12, `torch 2.14.0+cpu`,
`transformers 4.57.6`, float32, `CUDA_VISIBLE_DEVICES=-1`, uncommitted working tree on branch
`fix/qwen3-embedding-tutorial-review-2026-10-17`, 2026-10-18 (its package logic is that of commit `a93f99e`; the
edits made before that commit were docstring and line wrapping only). Not a hosted run;
timings are on a shared machine.

| Config | Layers | LR | Best epoch | Val recall@1 | Val MRR | Train + val s |
|---|---|---|---|---|---|---|
| C0 | 2 | 5e-05 | 2 | 0.7273 | 0.8284 | 417 |
| C1 | 2 | 2e-05 | 2 | 0.6818 | 0.7962 | 518 |
| C2 | 4 | 2e-05 | 2 | 0.7208 | 0.8270 | 554 |

Frozen validation MRR 0.6991. Selection: **C0** (no challenger beat C0 by the margin); selected model
`adapted C0`. Retraining the selected configuration reproduced the grid row exactly (same
per-epoch validation MRR).

| Split (CPU, float32) | Frozen recall@1 | Frozen MRR | Adapted recall@1 | Adapted MRR |
|---|---|---|---|---|
| Validation (154) | 0.5844 | 0.6991 | 0.7273 | 0.8284 |
| Exposed development test (385) | 0.6338 | 0.7422 | 0.8026 | 0.8785 |

Exposed development test, paired MRR difference 0.1363 (bootstrap 95 % CI 0.1069 to
0.1664; 115 queries improved, 21 worse); lexical baseline recall@1
0.3065. Old → new on the exposed development test: the selected configuration is
C0, so the adapted numbers are those of the old default's configuration retrained under the same fixed settings
(old CPU build record at `9bcb449`: frozen 63.4 % → adapted 80.3 % recall@1, MRR 0.742 → 0.879).

**Transfer check (CLINC150, display only).** 300 queries / 150 intent phrases, digest `2f45326a…`:
frozen recall@1 0.6600, MRR 0.7608; adapted recall@1 0.6667,
MRR 0.7641; paired MRR difference 0.0033 (95 % CI -0.0143 to 0.0217);
lexical recall@1 0.2933, random-floor MRR 0.0373. CPU stage times: frozen
evaluation 72.6 s, grid 1488.3 s, selected retrain 372.1 s, adapted evaluation
72.5 s, transfer 51.0 s.

**Open:** hosted BYOD (REL12). Status stays **Candidate**. *(2026-10-17: the hosted T4 run of the regenerated blob, which was open here, is recorded below as the evaluation of record.)*

### 2026-10-18 peer-review follow-up (row 72, F1-F14)

The peer review of PR #14 traced every path that can score the fresh test and found only Section 8, under
`preregistered`. Round-2 changes: the transfer check is split by CLINC domain (F1); the keep-frozen rule no longer runs
into the adapted-above-frozen stop, the frozen model is then the system scored on the fresh test, and a rejected adapter
is written with a `_not_selected` tag (F2); an intent-level cluster bootstrap is printed beside the pre-registered
query-level interval, as a disclosed secondary analysis that leaves the verdict rule unchanged (F4); grid records carry
run counters, and a selection that differs from the first pre-registered grid of the runtime is exploratory (F5);
`fresh_result` and the evaluation report record device and dtype (F12). The evaluation of record must be a hosted
**T4, bfloat16** run with the pre-registered settings; a CPU float32 pass is labelled the same way in-notebook but is not
the record.

**Deviation from the pre-registration's time estimate (F7).** The pre-registration estimated the CPU grid at 10-15 min;
the 2026-10-17 build record measured 1,488 s (24.8 min; 417-554 s per configuration) on a shared workstation. The
pre-registration text stays as committed; the notebook's timings now quote the measurement.

**Fresh-test exposure through the workshop (F8).** The semantic-search workshop draws 2 test.csv messages per intent with
its `SAMPLE_SEED` form field. At seed 42 (every recorded workshop run) none is a fresh-test message; at seeds 0, 1, 7 and
123 the draw holds 36, 29, 41 and 46 of them (checked 2026-10-18 with `samples.workshop_w16_queries`, data only). Maintainer
runs of the workshop keep seed 42; any other run is to be recorded here as a fresh-test exposure. The workshop is unchanged.

**REL13: CPU development look — transfer check split (F1), CPU build record, added after the 2026-10-17 record.** The adapted model is C0, retrained
on CPU with the pre-registered settings (per-epoch validation MRR identical to the grid row). The fresh test was masked.
Each row ranks against its own intent phrases. The pooled row reproduces the 2026-10-17 numbers exactly.

| Row (CPU, float32) | Queries / phrases | Frozen MRR | Adapted MRR | Delta MRR | 95 % CI, queries | 95 % CI, intents |
|---|---|---|---|---|---|---|
| Near-domain (CLINC banking + credit_cards) | 60 / 30 | 0.7238 | 0.7833 | +0.0595 | 0.0190 to 0.1082 | 0.0195 to 0.1085 |
| Other 8 domains | 240 / 120 | 0.7946 | 0.7736 | -0.0209 | -0.0418 to -0.0017 | -0.0460 to 0.0026 |
| Pooled (2026-10-17 view) | 300 / 150 | 0.7608 | 0.7641 | +0.0033 | -0.0143 to 0.0217 | -0.0170 to 0.0239 |

Old -> new: the pooled interval across 0 hid a near-domain gain and a small other-domain loss whose query-level interval
excludes 0 but whose intent-level interval does not. Display only; nothing in the selection changes. On the exposed
development test the intent-level interval of the paired MRR difference is 0.0877 to 0.1838 (query-level 0.1069 to
0.1664).

### 2026-10-17 hosted T4 evaluation of record (blob `971187e6`, PR #14 head `1cc43ed`)

**REL13: pre-registered evaluation of record.** This is the first hosted run of the pre-registered settings and the first
scoring of the fresh test by any run (the CPU checks above masked it). Later hosted runs with identical settings are
reproductions; a run with any other setting is exploratory and never scores the fresh test.

- **File.** [`docs/execution-evidence/2026-10-17-1cc43ed/`](execution-evidence/2026-10-17-1cc43ed/): executed notebook
  (SHA-256 `fa6a801d4dd305b69b0aa36be805316e97e09769c4ff5d80f8a9314770e3bbdf`), `exec.log` (`fe1ba286…`) and
  `run_summary.json` (`e411bcee…`), byte for byte (`-text`).
- **Source match.** The fetched notebook's `git hash-object` is `971187e6a67b878a6ce11c19c0b297c9d512d719`, the blob at
  `1cc43ed`, and it is byte-equal to the committed file. All 40 executed cells have the same ids and order, with no
  source differences and no `# @param` edits. `NOTEBOOK_SOURCE.repository_revision` `6a90df9`, `module_sha256`
  `00f2ef59…`, generator `build_notebook.py/2.2`, spec 2.5.
- **Executor.** Colab CLI 0.7.4 (`colab exec -f`) on a fresh Colab VM, Tesla T4 (session `suite-qwen3-1cc43ed-3e4c`).
  This is sequential CLI execution, not a browser Run all: forms and uploads were not exercised, and the file has no
  execution counts (order from `exec.log`, cells 1..14). Kernel Python 3.13.15; uv isolated environment CPython 3.12.12,
  47 locked packages, built in 44 s; `torch 2.14.0+cu130`, `transformers 4.57.6`; `cuda:0`, bfloat16.
- **Execution.** 14/14 code cells (the three carried-module cells print nothing), 0 error outputs, no restart, session wall
  382.9 s. 11 snapshot files (1,207,487,471 B) fetched from the Hub and digest-verified. Stage times: grid 138.9 s,
  selected retrain 47.7 s, transfer check 11.1 s.
- **Pre-registration.** `preregistered` True, `settings_differing` [], `test_look` "pre-registered evaluation (this
  runtime)", selected model `adapted C0`, keep-frozen False.

**Validation grid (154 queries; best epoch 2 for every configuration).**

| Config | Layers | LR | Val MRR by epoch 0 → 1 → 2 | Val recall@1 | Val MRR |
|---|---|---|---|---|---|
| C0 | 2 | 5e-05 | 0.6962 → 0.7664 → 0.7877 | 0.6753 | 0.7877 |
| C1 | 2 | 2e-05 | 0.6962 → 0.714 → 0.7197 | 0.5909 | 0.7197 |
| C2 | 4 | 2e-05 | 0.6962 → 0.719 → 0.7299 | 0.5909 | 0.7299 |

Frozen validation recall@1 0.5779, MRR 0.6962. Selection **C0** ("no challenger beat C0 by the margin": C1 −0.068 and C2
−0.058 against C0; C0 +0.0915 over frozen, so the adapter is kept). The selected retrain repeated the C0 curve exactly.
The grid summary line prints `'lr': 0.0001` for C0 and `'lr': 0.0` for C1/C2 because it rounds every float to four
decimals (`tools/notebook_template.py`, grid summary print); the configuration headers and the final configuration
(`5e-05`) are correct. Display only; open as a follow-up.

**Fresh test (770 queries, 10 per intent, manifest `fe68b43a…`), the evaluation of record.**

| System | Recall@1 | Recall@5 | Recall@10 | MRR | Median rank |
|---|---|---|---|---|---|
| Frozen | 0.6351 | 0.8688 | 0.9312 | 0.7338 | 1 |
| Adapted C0 | 0.761 | 0.9571 | 0.9779 | 0.8455 | 1 |

Paired MRR difference **+0.1118**, bootstrap 95 % interval **0.0911 to 0.134** (query-level, pre-registered) and
**0.0711 to 0.1569** (intent-level cluster bootstrap, secondary); 215 queries improved, 39 worse, 516 unchanged. Both
intervals exclude 0; verdict *improved*. Re-derived from the printed values: every recall × 770 is a whole count (489 →
586 at recall@1), each MRR lies inside the bounds its recall@1/5/10 allow, and 215 + 39 + 516 = 770. The bootstrap
intervals were not recomputed: the per-query ranks stayed on the released VM.

**Exposed development test (385; not independent — the old default was chosen on it).** Random floor 0.013 / 0.064,
lexical 0.3065 / 0.4159, frozen 0.6312 / 0.741, adapted 0.7662 / 0.8498 (recall@1 / MRR; Δ MRR +0.1087). This revision
prints no paired interval for this split.

**CLINC150 transfer check (display only).**

| Row (T4, bfloat16) | Queries / phrases | Frozen MRR | Adapted MRR | Delta MRR | 95 % CI, queries | 95 % CI, intents | Improved / worse |
|---|---|---|---|---|---|---|---|
| Near-domain (CLINC banking + credit_cards) | 60 / 30 | 0.7237 | 0.7464 | +0.0227 | −0.0097 to 0.0579 | −0.0100 to 0.0584 | 12 / 4 |
| Other 8 domains | 240 / 120 | 0.7940 | 0.7781 | −0.0159 | −0.0348 to 0.0022 | −0.0381 to 0.0062 | 24 / 26 |

Every interval includes 0. This revision prints no pooled 150-phrase row.

**Against the CPU development look (float32, fresh test masked).** The selection is the same (C0, best epoch 2 everywhere),
but the numbers are not close:

| Measure | CPU development look | Hosted T4 record | Difference |
|---|---|---|---|
| Frozen validation MRR / recall@1 | 0.6991 / 0.5844 | 0.6962 / 0.5779 | −0.0029 / −0.0065 |
| C0 validation MRR / recall@1 | 0.8284 / 0.7273 | 0.7877 / 0.6753 | −0.0407 / −0.0520 |
| C1 validation MRR / recall@1 | 0.7962 / 0.6818 | 0.7197 / 0.5909 | −0.0765 / −0.0909 |
| C2 validation MRR / recall@1 | 0.8270 / 0.7208 | 0.7299 / 0.5909 | −0.0971 / −0.1299 |
| C2 − C0 validation MRR | −0.0014 (inside the margin) | −0.0578 | C2 no longer near C0 |
| Exposed frozen recall@1 / MRR | 0.6338 / 0.7422 | 0.6312 / 0.741 | −0.0026 / −0.0012 |
| Exposed adapted recall@1 / MRR | 0.8026 / 0.8785 | 0.7662 / 0.8498 | −0.0364 / −0.0287 |
| Exposed Δ MRR | +0.1363 (CI 0.1069 to 0.1664) | +0.1087 (no CI printed) | −0.0276 |
| Transfer near-domain frozen / adapted MRR | 0.7238 / 0.7833 | 0.7237 / 0.7464 | −0.0001 / −0.0369 |
| Transfer near-domain Δ (CI queries) | +0.0595 (0.0190 to 0.1082, excludes 0) | +0.0227 (−0.0097 to 0.0579, includes 0) | −0.0368 |
| Transfer other-8 frozen / adapted MRR | 0.7946 / 0.7736 | 0.7940 / 0.7781 | −0.0006 / +0.0045 |
| Transfer other-8 Δ (CI queries) | −0.0209 (−0.0418 to −0.0017, excludes 0) | −0.0159 (−0.0348 to 0.0022, includes 0) | +0.0050 |
| Adapter size | 125,849,896 B (float32) | 62,926,128 B (bfloat16) | dtype |
| Grid / retrain / transfer time | 1,488.3 / 372.1 / 51.0 s | 138.9 / 47.7 / 11.1 s | device |

The frozen numbers agree to within 0.007 (bfloat16 versus float32 inference). The adapted numbers are 0.03–0.10 MRR lower
on T4, more so at the lower learning rate (C1, C2). The likely cause (inferred, not tested) is that training on CUDA
updates bfloat16 weights directly, so AdamW steps of 2e-5 to 5e-5 are partly lost to bfloat16 rounding. Neither CPU
transfer conclusion (near-domain gain, other-domain loss) is reproduced at T4 precision. None of this changes a
setting: the CPU look is disclosed, and the hosted run is the record.

**Against the previous hosted run (2026-10-09, blob `26a62b80`).** The exposed-test frozen and adapted metrics, the C0
validation curve, the lexical baseline, the random floor and the document self-cosine (median 0.8458, minimum 0.7634)
are equal at printed precision. Every other difference is a PR #14 change: the grid, fresh-test and transfer blocks, the
split labels, reload parity on validation (0.787748 both ways) instead of the test split, the extra
`…_grid_validation.json` export and the `started_from` text. The rest is timing.

**Evidence boundary.** Default path only, by CLI sequential execution. BYOD (REL12), the activity and the optional
experiments were not run. The `outputs/` files stayed on the released VM, so their bytes (per-query ranks, the
evaluation report, the transfer report) were not inspected.

| Journey | Verdict |
|---|---|
| Clean default path (isolated env, no restart; CLI sequential) | PASS |
| Activity / optional experiments | not assessed in this run |
| BYOD | not assessed in this run |

Status stays **Candidate**.

### 2026-10-18 pre-registration amendment 1: float32 recipe and fresh test 2

Owner decision (Kurt, 2026-10-18): keep the record, fix the GPU training precision, and pre-register a second fresh test.
`docs/qwen3-embedding-preregistration-amendment-2026-10-18.md` was committed alone first (`49d82f7`), before any code
change of this round.

- **Recipe 1 record unchanged.** The hosted T4 run of `1cc43ed` (blob `971187e6`, 2026-10-17) remains the
  pre-registered evaluation of record for recipe 1 (bf16 CUDA training): fresh-test MRR 0.7338 -> 0.8455, +0.1118, 95 %
  CI 0.0911 to 0.134 (queries), 0.0711 to 0.1569 (intents). `samples.RECORDED_EVALUATIONS` holds it under recipe 1.
- **Finding.** `pipeline.py` loaded bfloat16 on CUDA and AdamW stepped the bfloat16 parameters with no float32 master
  copy; T4 validation MRR C0 0.788 vs CPU float32 0.828 (C1 0.720 vs 0.796, C2 0.730 vs 0.827).
- **Recipe 2.** float32 on every device (`MODEL_DTYPE`, `model_dtype`); `adapt` refuses non-float32 trainable
  parameters and checks that the AdamW moments are float32; adapter manifests record their dtype (float32: 125,849,896 B
  for 2 layers). Chosen over bf16 or fp16 autocast because a T4 has no bf16 tensor cores and fp16 risks overflow in a
  bf16-trained decoder; memory well under 16 GB. **T4 estimate (not measured):** grid 2.5-5 min, whole session 7-12 min.
- **Fresh test 2.** 770 (10 per intent) from the 1,770 test.csv messages no run has scored (excluding the exposed 385,
  W16's 154, the 22 train repeats and fresh test 1's 770; 19-24 per intent), `Random(20261018)`, manifest
  `cd0171285dfd6ccc72b88d82b43ad856af6869a84a49cc50ac06046119f37a63`; near-duplicates reported: 12 in the draw, 29 in the
  pool. Only fresh test 2 is scored as the pre-registered evaluation; fresh test 1 is shown every run as
  "fresh test 1 (scored once, 2026-10-17, bf16 recipe; now an exposed test)".
- **Recipe-2 record (REL13):** the hosted T4 run of `9ca75dc` (blob `de48070a`, 2026-10-18, cuda float32) — section
  *2026-10-18 hosted T4 evaluation of record, recipe 2* below. `RECORDED_EVALUATIONS[recipe 2]` is still `None` at this
  blob and is a pending code follow-up (requote), so a later identical run still prints "pre-registered evaluation (this
  runtime)"; this file names the run of record. *(Was: none yet; the first hosted pre-registered T4 run of recipe 2 will
  be THE evaluation of record for it.)*
- **CPU check (both fresh tests masked; a guard aborted on any fresh-test query).** CPU was already float32, so the grid
  was not re-run. Smoke on real weights: parameters float32; a 64-pair, 1-epoch C0 adaptation passed both float32
  checks (validation MRR 0.6991 -> 0.7152; the frozen value equals the 2026-10-17 CPU record); manifest dtype float32,
  125,849,896 B; reload float32 with identical validation MRR (0.715178 both). Not a hosted run.

### 2026-10-18 hosted T4 evaluation of record, recipe 2 (blob `de48070a`, PR #14 head `9ca75dc`)

**REL13: pre-registered evaluation of record for recipe 2 (float32 on every device; fresh test 2).** This is the first
hosted run of the recipe-2 pre-registered settings (amendment 2026-10-18) and the first scoring of fresh test 2 by any
run (the CPU check of the amendment masked both fresh tests). Later hosted runs with identical recipe-2 settings are
reproductions. The recipe-1 record (2026-10-17, `1cc43ed`, section above) stays the evaluation of record for recipe 1
and is not replaced; on this run fresh test 1 is an **exposed test**, shown for continuity.

- **Files.** [`docs/execution-evidence/2026-10-18-9ca75dc/`](execution-evidence/2026-10-18-9ca75dc/): executed notebook
  (SHA-256 `5e4be566f847e0abf6e1646ea5197d78b3199b7ef21fc26670ed2b3bbe3a1843`), `exec.log` (`5040de3b…`) and
  `run_summary.json` (`b906a644…`), byte for byte (`-text`).
- **Source match.** The fetched notebook's `git hash-object` is `de48070a3621ed61529948a1b1f3a0e671bf9ed4`, the blob at
  `9ca75dc`, and it is byte-equal to the committed file. All 40 executed cells have the same ids and order, with no source
  differences and no `# @param` edits. `NOTEBOOK_SOURCE.repository_revision` `f7f3dc0`, `module_sha256` `cfa5687b…`,
  generator `build_notebook.py/2.2`, spec 2.5.
- **Executor.** Colab CLI 0.7.4 (`colab exec -f`) on a fresh Colab VM, Tesla T4 (session `suite-qwen3-9ca75dc-3e4c`,
  started 2026-10-18 00:38:29 UTC). Sequential CLI execution, not a browser Run all: forms and uploads were not exercised,
  and the file has no execution counts (order from `exec.log`, cells 1..14). Kernel Python 3.13.15; uv isolated
  environment CPython 3.12.12, 47 locked packages, built in 51 s; `torch 2.14.0+cu130`, `transformers 4.57.6`; `cuda:0`,
  **float32** (printed in the fresh-test block).
- **Execution.** 14/14 code cells (the three carried-module cells print nothing), 0 error outputs, no restart, session wall
  346.8 s. 11 snapshot files (1,207,487,471 B) fetched from the Hub and digest-verified. Grid 105.0 s, selected re-train
  35.1 s, transfer check 8.3 s.
- **Pre-registration.** `preregistered` True, `settings_differing` [], `test_look` "pre-registered evaluation (this
  runtime)", selected model `adapted C0`.

**Validation grid (154 queries; best epoch 2 for every configuration).**

| Config | Layers | LR | Val MRR by epoch 0 → 1 → 2 | Val recall@1 by epoch | Val MRR |
|---|---|---|---|---|---|
| C0 | 2 | 5e-05 | 0.6991 → 0.8118 → 0.8284 | 0.5844 → 0.7013 → 0.7273 | **0.8284** |
| C1 | 2 | 2e-05 | 0.6991 → 0.7674 → 0.7962 | 0.5844 → 0.6364 → 0.6818 | 0.7962 |
| C2 | 4 | 2e-05 | 0.6991 → 0.8077 → 0.827 | 0.5844 → 0.7013 → 0.7208 | 0.827 |

Selection **C0** ("no challenger beat C0 by the margin": C1 −0.0322 against C0; C2 −0.0014 against the best 2-layer
configuration, inside the 0.02 margin; C0 +0.1293 over frozen, so the adapter is kept). The selected re-train repeated the
C0 curve (recall@5 0.8701 → 0.9675 → 0.974, train loss 0.4788 → 0.1775).

**Fresh test 2 (770 queries, 10 per intent, manifest `cd017128…`), the recipe-2 evaluation of record.**

| System | Recall@1 | Recall@5 | Recall@10 | MRR | Median rank |
|---|---|---|---|---|---|
| Frozen | 0.613 | 0.8701 | 0.9312 | 0.7274 | 1 |
| Adapted C0 | 0.7727 | 0.9675 | 0.9844 | 0.8595 | 1 |

Paired MRR difference **+0.1321**, bootstrap 95 % interval **0.111 to 0.1547** (query-level, pre-registered) and
**0.0886 to 0.1748** (intent-level, secondary); 240 queries improved, 44 worse, 486 unchanged. Both intervals exclude 0;
verdict *improved*. Re-derived from the printed values: every recall × 770 is a whole count (frozen 472 / 670 / 717,
adapted 595 / 745 / 758); each MRR lies inside the bounds its recall@1/5/10 allow over 77 documents; 240 + 44 + 486 = 770;
the delta is the difference of the printed means; the intent interval is the wider one. The bootstrap was not re-run: the
per-query ranks stayed on the released VM.

**Fresh test 1 (770, manifest `fe68b43a…`; now an exposed test, scored under recipe 1 on 2026-10-17).** Frozen 0.6338 /
0.8714 / 0.9312 / MRR 0.7324, adapted 0.7857 / 0.9714 / 0.9831 / MRR 0.8666; paired Δ MRR +0.1342, 95 % CI 0.1113 to
0.1588 (queries; no intent-level interval printed for this split). Not independent evidence: it is the recipe-1 record's
test.

**Exposed development test (385; not independent).** Random floor 0.013 / 0.064, lexical 0.3065 / 0.4159, frozen
0.6338 / 0.7422 (recall@5 0.8857, recall@10 0.9403), adapted 0.8026 / 0.8785 (0.9792, 0.9948) (recall@1 / MRR;
Δ recall@1 +0.1688, Δ MRR +0.1363). No paired interval is printed for this split on this revision.

**CLINC150 transfer check (display only).**

| Row (T4, float32) | Queries / phrases | Frozen MRR | Adapted MRR | Delta MRR | 95 % CI, queries | 95 % CI, intents | Improved / worse |
|---|---|---|---|---|---|---|---|
| Near-domain (CLINC banking + credit_cards) | 60 / 30 | 0.7238 | 0.7833 | +0.0595 | 0.0190 to 0.1082 | 0.0195 to 0.1085 | 14 / 3 |
| Other 8 domains | 240 / 120 | 0.7946 | 0.7736 | −0.0209 | −0.0418 to −0.0017 | −0.0460 to 0.0026 | 22 / 31 |

The near-domain gain excludes 0 at both levels; the other-domain loss excludes 0 at query level only. This revision prints
no pooled 150-phrase row.

**Did float32 close the gap?** Yes. Every validation, exposed-test and transfer number the CPU float32 record printed is
reproduced on the T4 at printed precision; recipe 1 on the T4 was 0.03–0.10 MRR below.

| Measure | CPU float32 record (2026-10-17/18) | Recipe 1, hosted T4 bf16 (2026-10-17) | Recipe 2, hosted T4 float32 (this run) |
|---|---|---|---|
| Frozen validation MRR / recall@1 | 0.6991 / 0.5844 | 0.6962 / 0.5779 | 0.6991 / 0.5844 |
| C0 / C1 / C2 validation MRR | 0.8284 / 0.7962 / 0.8270 | 0.7877 / 0.7197 / 0.7299 | 0.8284 / 0.7962 / 0.827 |
| C2 − C0 validation MRR | −0.0014 | −0.0578 | −0.0014 |
| Selection | C0 | C0 | C0 |
| Exposed frozen recall@1 / MRR | 0.6338 / 0.7422 | 0.6312 / 0.741 | 0.6338 / 0.7422 |
| Exposed adapted recall@1 / MRR | 0.8026 / 0.8785 | 0.7662 / 0.8498 | 0.8026 / 0.8785 |
| Exposed Δ MRR | +0.1363 (CI 0.1069 to 0.1664) | +0.1087 | +0.1363 (no CI printed) |
| Fresh test 1 frozen / adapted MRR | masked | 0.7338 / 0.8455 (record, recipe 1) | 0.7324 / 0.8666 (exposed) |
| Fresh test 1 Δ MRR (CI queries) | masked | +0.1118 (0.0911 to 0.134) | +0.1342 (0.1113 to 0.1588) |
| Fresh test 2 | masked | not drawn | 0.7274 → 0.8595, +0.1321 (record, recipe 2) |
| Transfer near-domain Δ (CI queries) | +0.0595 (0.0190 to 0.1082) | +0.0227 (−0.0097 to 0.0579) | +0.0595 (0.0190 to 0.1082) |
| Transfer other-8 Δ (CI queries) | −0.0209 (−0.0418 to −0.0017) | −0.0159 (−0.0348 to 0.0022) | −0.0209 (−0.0418 to −0.0017) |
| Transfer pooled 300 / 150 | +0.0033 (−0.0143 to 0.0217) | not printed | not printed |
| Reload parity, validation MRR | — | 0.787748 both | 0.828415 both |
| Document self-cosine min / median | — | 0.7634 / 0.8458 | 0.5197 / 0.6797 |
| Adapter size | 125,849,896 B | 62,926,128 B | 125,849,896 B |
| Grid / retrain / transfer time | 1,488.3 / 372.1 / 51.0 s | 138.9 / 47.7 / 11.1 s | 105.0 / 35.1 / 8.3 s |
| Session wall | — | 382.9 s | 346.8 s |

The only differences from the CPU record are timings and the numbers CPU did not print (fresh tests masked; reload parity
and self-cosine). The frozen T4 numbers moved from recipe 1 to recipe 2 by at most 0.0065 recall@1 and 0.0029 MRR
(bfloat16 against float32 inference). The amendment's T4 estimate (grid 2.5–5 min, whole session 7–12 min, not measured) was high: the grid
took 105.0 s and the session 346.8 s (5.8 min). Peak GPU memory is not printed by the notebook and was not measured.

**Evidence boundary.** Default path only, by CLI sequential execution. BYOD (REL12), the activity and the optional
experiments were not run. The `outputs/` files stayed on the released VM, so their bytes (per-query ranks, the evaluation
report, the transfer report) were not inspected.

| Journey | Verdict |
|---|---|
| Clean default path (isolated env, no restart; CLI sequential) | PASS |
| Activity / optional experiments | not assessed in this run |
| BYOD | not assessed in this run |

Status stays **Candidate**.

## Qwen3 semantic search and reranking notebook — 2026-09-26 follow-up

This record concerns `DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb` on PR #8, separately from the primary E2E embedding tutorial above. Assessment used fleet NOTEBOOK_SPEC 2.2 (blob `046d866eac7cc67b1539a4ba370ebc965341c87b`) and this repository's search/reranking specification. The compatible 2.1 declaration remains; the 2.2 guided recommendations do not change mandatory profile semantics.

Confirmed fixes:

- GDL7–12: added stage-specific predictions, expected-result guidance, worked coverage/conditional-metric checkpoints and collapsed infrastructure. Extended metric definitions for beginners.
- GDL10: the learner activity now runs only the existing optional K-sweep, preserving canonical K/rankings/metrics. It explicitly treats the reused test queries as exploratory, requiring a separate untouched set for a tuning claim.
- BYOD: partial labels no longer silently suppress all evaluation; mixed labels and unknown gold IDs fail before model reload.
- BYOD: release the canonical reranker before loading the BYOD embedder, then release that embedder before reloading the reranker.
- BYOD: persist optional rankings and results separately, with file digests, effective K, instructions, model revisions and runtime. The terminal summary checks those files when BYOD is enabled.

Local baseline: release validator passed and 10/10 comparative-notebook tests passed. Revised checks: 17/17 focused tests pass, generator parity/release validator pass, and changed Python files pass Ruff. Seven added tests execute the actual notebook optional-path cells with deterministic model doubles. They cover labelled/unlabelled exports, invalid-label refusal before reload, one-model residency, preservation of canonical output, a depth-sweep example with a shortlist miss, and failed embedding load/computation followed by a successful retry. Failure cleanup clears retained model traceback frames before restoring the reranker. These checks validate control flow and metrics, **not** real checkpoint inference, GPU memory usage, or full REL12 qualification.

```powershell
$env:PYTHONPATH='src'
python -m pytest --noconftest -o addopts= tests/test_semantic_search_reranking_workshop.py tests/test_workshop_optional_paths.py
python tools/validate_release_assets.py
python tools/build_semantic_search_reranking_workshop.py --check
```

### Open release gates

1. **Uninterrupted fresh-runtime Run all is not established.** The current installer updates packages in the notebook kernel. When Colab has pre-imported a replaced package, it deliberately raises and asks for a manual session restart. A restart-assisted run must be labelled as such and does not close the no-interaction gate. Isolated model execution or another verified bootstrap design is still needed for hosts exhibiting this mismatch; deleting stale modules from `sys.modules` is not a safe fix.
2. Record an exact-revision supported-runtime run, including commit/blob, package/device inventory, bootstrap/restart actions, all five canonical exports and their digests. No such run was performed for this follow-up.
3. Run real BYOD with representative labelled documents/queries through both models and inspect both exports. Repeat with unlabelled queries and verify `not-measurable` plus rankings; reject mixed labels and unknown gold IDs before BYOD model loading. Retain input digests and run evidence. Character-valid text can exceed the token budget; assess truncation on representative data.
4. Run the optional sweep with real models and retain its exploratory results separately from the canonical K=6 claim. Measure memory/timing rather than inferring either from test doubles.

Status remains **Candidate**. The primary embedding tutorial's existing release-grade evidence is not inherited by this composed notebook. The manual-restart gate and hosted/full-model BYOD evidence prevent a gold-standard claim.


### Colab NumPy setup failure — 2026-09-26

The maintainer-supplied run stopped in setup before model execution: NumPy 2.1.3 was already loaded, while the notebook installed 2.5.3. The [failure record](execution-evidence/2026-09-26/colab-setup-failure.json) records the independently inspected error. The supplemental notebook retains an already loaded NumPy 2.x, integrating the concurrent host-preservation fix, and uses 2.1.3 as the fallback pin when NumPy is not yet loaded. The observed Colab 2.1.3 is preserved instead of replaced. Other model/runtime pins are unchanged; stale-module detection remains enabled. Declared upstream requirements permit 2.1.3 (Transformers and datasets require >=1.17; the closed-set SciPy pin requires >=2.0,<2.8).

A regression executes the real setup prefix against a simulated Colab preloaded NumPy and package installer: it reproduces the original restart error before the fix and completes without a restart after it. This is setup regression evidence, not a full model/Colab rerun. A new hosted Run all is still required to discover any downstream issues. Use a fresh runtime for that rerun; the prior failed session already replaced installed packages.


### Maintainer-supplied successful Colab run — 2026-09-26

This later record supersedes the earlier default-path setup-failure status for the supplemental notebook. The maintainer supplied the [executed notebook](execution-evidence/2026-09-26/DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb) and authorized merging PR #8. The file is preserved byte-for-byte (SHA-256 `a2d8358e75f0a9c9ab9416d640977f2fb2566d45cd33aabaccc89e58338391ab`). All 19 code cells have execution counts, with 45 saved outputs and zero saved error outputs. All executable Python ASTs match commit `2ff64ec616773509159abe0aed33ec81e8a857da`, tutorial blob `5512be3076a1af0b7a7ff1e6fb0c3b939871162f`; Colab added only `# @title` comments to 18 cells. The tutorial source itself is unchanged by this evidence commit.

The run uses the canonical settings: USE_BYOD=False, RUN_K_SWEEP=False, seed 42, K=6; 154 queries and 77 documents. Saved runtime inventory: Python 3.13.15, torch 2.14.0+cu130, Transformers 4.57.6, huggingface-hub 0.36.2, NumPy 2.1.3 (preloaded by host), cuda:0 and bfloat16. Setup proceeds through model loading, scoring and final export verification. Five required canonical files are reported written and checked; the external files themselves were not supplied, so their bytes/digests were not independently inspected.

Saved results: lexical recall@1 0.3117; embedding recall@1 0.6104; shortlist coverage@6 0.9156; two-stage recall@1 0.6234 and MRR 0.7379. Reranking helped 29 queries, hurt 24, left 88 unchanged and could not recover 13 shortlist misses. This is evidence for the default composed retrieval run, not for optional BYOD or K-sweep paths.

The supplied artifact does not independently establish runtime freshness or the absence of manual reruns/restarts; no such claim is inferred from saved execution counts. This merge records the successful run and maintainer approval without declaring all gold-standard/REL12 gates closed. Real-model BYOD and optional sweep qualification remain pending.


### 2026-10-03 uv isolated environment

The setup cell of `DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb` no longer installs into the notebook kernel. It downloads a size- and SHA-256-pinned `uv` 0.12.15 wheel, creates a managed CPython 3.12.12 environment, installs `tutorials/requirements-semantic-search-workshop.lock.txt` into it with `--require-hashes --only-binary :all:`, and runs every later code cell in one persistent worker process there. The restart guard is gone. The direct pins are unchanged (torch 2.14.0, torchvision 0.29.0, torchaudio 2.11.0, Transformers 4.57.6, huggingface-hub 0.36.2, safetensors 0.8.0, NumPy 2.1.3); the 2026-09-26 run's kernel Python was 3.13.15, while the isolated environment uses 3.12.12. The notebook now needs a Linux x86_64 runtime. This addresses open release gate 1 above in design only.

Tutorial blob `99a726dcb8bd92f4a5ff0dd5fbc190b6a070629f` (origin/main `8c87fc6`) → `d72ea820c305724b27971eac6aff11a04e07f158`. No hosted run of the old blob `99a726dc` is recorded here; the latest recorded hosted run is the 2026-09-26 maintainer-supplied Colab run of blob `5512be30`, whose default results remain the reference. Local checks (pytest, Ruff, the release validator, both generator `--check`s, and the real worker process on Linux CPython 3.12 with a stand-in interpreter) are not clean-runtime evidence. A hosted re-run of the new blob is pending. Status: **Candidate**.


### Maintainer-supplied Colab execution of blob `d72ea820` — 2026-10-04

- **File.** [executed notebook](execution-evidence/2026-10-04/DIMER_Qwen3_Semantic_Search_Reranking_Workshop_2996bb2_colab-browser-t4.ipynb), preserved byte-for-byte, SHA-256 `5c87a5f9b68792e0b95bded386bd9555e3431ce942042806cbf5b5d23cf42c7f`.
- **Source match.** All 47 cells have the same ids and order as the notebook at PR #10 head `2996bb2` (blob `d72ea820c305724b27971eac6aff11a04e07f158`), with zero source differences, including no `# @param` or `# @title` edits.
- **Executor.** Google Colab (browser, maintainer-run), Tesla T4 (`gpuType` T4). The kernel ran Python 3.13.15. The uv isolated environment built in 65 s: CPython 3.12.12, 47 locked packages, `torch 2.14.0+cu130`, Transformers 4.57.6, huggingface-hub 0.36.2 and NumPy 2.1.3 (pinned install), on `cuda:0` in bfloat16. No session restart was needed.
- **Execution.** 19/19 code cells ran, with execution counts 1..19 in order and 0 error outputs. Total wall time is not recorded because the saved file has no per-cell timing. Peak GPU memory is not printed by this notebook.
- **Results.** These match the 2026-09-26 run of blob `5512be30`:

  | Metric | 2026-09-26 | 2026-10-04 |
  |---|---|---|
  | Embedding recall@1 / MRR | 0.6104 / 0.7310 | 0.6104 / 0.7310 |
  | Shortlist coverage@6 | 0.9156 | 0.9156 |
  | Two-stage recall@1 / MRR | 0.6234 / 0.7379 | 0.6234 / 0.7379 |
  | Helped / hurt / unchanged / unrecoverable | 29 / 24 / 88 / 13 | 29 / 24 / 88 / 13 |
  | Lexical recall@1 (ties against gold) | 0.3117 | 0.3117 (diagnostic row) |
  | Lexical recall@1 (displayed ranking) | — | 0.3506 |

  The case-study cosine and rerank scores are also identical to 4 decimal places.
- **Why the outputs differ from the 2026-09-26 run.** Every difference traces to a source change between `5512be30` and `d72ea820`.
  - From `7f8b28a` (QSR-01/02, blob `99a726dc`, never hosted-run):
    - The lexical baseline now uses the displayed-ranking tie policy (0.3506). The former figure is kept as the named diagnostic row (0.3117, unchanged), plus a tie count of 93/154.
    - Gold and candidate phrases are printed, along with the gold-rank and first-stage-rank lines.
    - Two lookup CSVs are exported, so 7 required files instead of 5.
  - From `2996bb2` (uv move): the environment printout changed, and the in-kernel install and Hub progress/token-warning output is gone.
  - The remaining differences are timings only.
- **Evidence boundary.** This covers the default path only. BYOD and the optional K-sweep were not run, so they are not assessed in this run. The run closes open release gate 1 (an uninterrupted Run all with no restart) for the default path on Colab T4. The output files themselves were not supplied, so their bytes were not inspected.

| Journey | Verdict |
|---|---|
| Default Run all (isolated env, no restart) | PASS |
| Optional K-sweep | not assessed in this run |
| BYOD | not assessed in this run |

Status stays **Candidate**. Gates 3 and 4 above (real-model BYOD and the optional sweep) remain open.

### Colab CLI execution of blob `089696de` (PR #13) — 2026-10-14

- **File.** [executed notebook](execution-evidence/2026-10-14/DIMER_Qwen3_Semantic_Search_Reranking_Workshop_7dbcc38_colab-cli-t4.ipynb), preserved byte-for-byte, SHA-256 `71799c6e1610ed315e0a33707cac35afa7de5efdba83a950aca5fe4127fa33f6`. The run's `exec.log` (SHA-256 `48d6efdabbd5900d4b27c97bcb549b84c1eb6839d208758f345e577bd7aa82a9`) and `run_summary.json` (`452eb87d02579e80ba75c4ae3fc69401769417c318977fb2e2fecd403cc54bd9`) are kept in the maintainer's workspace.
- **Source match.** The fetched notebook's `git hash-object --no-filters` is `089696dec47ebf230d1f546c9f2ddd29ec0774c1`, the blob at PR #13 head `7dbcc38`. All 47 executed cells have the same ids and order as that blob, with no source differences and no `# @param` edits.
- **Executor.** Colab CLI 0.7.4 (`colab exec -f`) on a fresh Colab VM with a Tesla T4. This is sequential CLI execution, not a browser Run all: forms and uploads were not exercised, and the file has no execution counts (order comes from `exec.log`, cells 1..19). Kernel Python 3.13.15; the uv isolated environment built in 52 s (CPython 3.12.12, 47 locked packages, `torch 2.14.0+cu130`, Transformers 4.57.6). No restart was needed.
- **Execution.** 19/19 code cells ran, 0 error outputs, session wall 164.5 s. The short wall time is real work, not a cache: both snapshots were fetched from the Hub (11 and 13 files, each about 1.21 GB) and digest-verified; the embedder built the 77-document index on `cuda:0` in bfloat16 (6.77 s) and embedded 154 queries (3.79 s); the reranker loaded on `cuda:0` and scored 924 pairs (36.3 s). Every metric line printed.
- **Results.** Every computed number equals the 2026-10-04 browser run of `d72ea820`:

  | Metric | 2026-10-04 | 2026-10-14 |
  |---|---|---|
  | Lexical recall@1 (displayed / ties against gold) | 0.3506 / 0.3117 | 0.3506 / 0.3117 |
  | Embedding recall@1 / MRR over 77 | 0.6104 / 0.7310 | 0.6104 / 0.7310 |
  | Shortlist coverage@6 | 0.9156 | 0.9156 |
  | Two-stage recall@1 / MRR@6 | 0.6234 / 0.7379 | 0.6234 / 0.7379 |
  | Helped / hurt / unchanged / unrecoverable | 29 / 24 / 88 / 13 | 29 / 24 / 88 / 13 |
  | Embedding top-6 only MRR@6 | — | 0.7241 |
  | Paired top-1 (two-stage only / embedding only / both / neither) | — | 17 / 15 / 79 / 43, exact sign test p = 0.86 |
  | Rank direction sign test (covered queries) | — | p = 0.58 |
  | Query-text audit (repeated / conflicting) | — | 0 / 0 of 154 |

  The case-study cosine and rerank scores are identical to 4 decimal places.
- **Why the outputs differ from the 2026-10-04 run.** Every non-timing difference comes from PR #13's generator changes: the query-text duplicate audit line, the "Embedding top-6 only" reference row, the "MRR over 77" / "MRR@6" column note, the paired-comparison block, and the relabelled summary lines ("Rerank depth: 6 (the canonical run's configuration)", "Lexical (Jaccard) recall@1", "Embedding MRR over 77", "Embedding MRR@6 (top-6 only)", "Two-stage MRR@6", "Paired top-1 changes"). The remaining differences are timings (setup 52 s vs 65 s, index 6.77 s vs 11.32 s, query embedding 3.79 s vs 4.00 s, rerank 36.3 s vs 37.2 s) and the worker pid.
- **Evidence boundary.** Default path only, by CLI sequential execution. BYOD and the optional K-sweep were not run. The output files were written inside the VM and not downloaded, so their bytes (including the new `paired_comparison` block in `metrics.json`) were not inspected.

| Journey | Verdict |
|---|---|
| Default path (isolated env, no restart; CLI sequential) | PASS |
| Optional K-sweep | not assessed in this run |
| BYOD | not assessed in this run |

Status stays **Candidate**. Gates 3 and 4 above (real-model BYOD and the optional sweep) remain open.
