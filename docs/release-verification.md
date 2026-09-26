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
| 2026-09-19 | `ec6dd95` / `f3475b93` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-qwen3-embedding` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 323.6 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 26 files, 1209 MB staged from the Hub into a clean cache; comparison {recall@1: {random_floor: 0.013, lexical: 0.3065, frozen: 0.6312, adapted: 0.7662}, recall@5: {random_floor: 0.0649, lexical: 0.5039, frozen: 0.8857, adapted: 0.9714}, recall@10: {random_floor: 0.1299, lexical: 0.6753, frozen: 0.9377, adapted: 0.9896}, mrr: {random_floor: 0.064, lexical: 0.4159, frozen: 0.741, adapted: 0.8498}, median_rank: {lexical: 5, frozen: 1, adapted: 1}, delta_vs_frozen: {recall@1: 0.1351, recall@5: 0.0857, recall@10: 0.0519, mrr: 0.1087}}; reload parity {query_vectors_identical: True, mrr_in_memory: 0.8125, mrr_reloaded: 0.8125}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-qwen3-embedding/v3/evidence/` in the workspace |
| 2026-09-19 | `9bcb449` / `40e0a136` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 11 entries because the snapshot was pre-staged → `verify_snapshot` 11 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its digest checks → 10,003 + 3,080 rows read, 616 / 154 / 385 drawn over 77 intents with `check_split_disjoint` clean, 77 documents and digests `dacba395…` / `3d07cfd5…` / `df00c83e…` → four dataset refusals → input manifest with the oversized-batch refusal → `embed` of the 77 documents (0.72 s) and three test queries (0.18 s; 30 / 48 / 36 tokens) with all five sanity checks `True` → frozen top-3 lists → random floor → lexical baseline → frozen evaluation → `adapt` → validation + test evaluation → retrievals after adaptation → adapter export → reload parity) | 226.9 s | **PASSED** — 11/11 code cells; random floor recall@1 1.3 % / MRR 0.064; lexical baseline recall@1 30.6 %, recall@5 50.4 %, MRR 0.416, median rank 5; frozen test recall@1 63.4 %, recall@5 88.6 %, recall@10 94.0 %, MRR 0.742, median rank 1 (30.7 s); `adapt` 31,461,888 of 595,776,512 params, 616 pairs over 77 documents, 2 epochs, 133.3 s, validation MRR 0.699 → 0.812 → 0.828 (recall@1 58.4 → 70.1 → 72.7 %; `best_epoch` 2, train loss 0.479 → 0.178); **adapted test recall@1 80.3 %, recall@5 97.9 %, recall@10 99.5 %, MRR 0.879 (Δ +16.9 / +9.4 / +5.5 points, +0.136 MRR)**; the three probe queries ranked their gold intent first before and after (the top-3 below the gold changed for all three); document self-cosine to the frozen vectors median 0.680, minimum 0.520; per-batch report `not-measurable`; adapter 125,849,896 B / 22 tensors, SHA-256 `ac38839e…`; reload parity exact (query vectors identical, 77-query MRR 0.859199 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-14 | `cbeec85` / `7715612eea32` (`TASK-INFERENCE`, superseded) | Kaggle T4 (`kurtvalcorza/dimer-nb2-qwen3-embedding` v2) | Default sample path of the inference-only notebook: the four README sentences, `stage_missing_files` fetching `model.safetensors` from the Hub, `verify_snapshot` over 11 files, query/document embedding with the cosine check, `not-measurable` report, CSV + JSON exports | 234.2 s | **PASSED** — 8/8 code cells, 1,207 MB staged; history only |

## Primary embedding tutorial status

**Release-grade.** The `E2E` notebook blob `f3475b93` (committed at `ec6dd95`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 323.6 s, 26 files, 1209 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

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
