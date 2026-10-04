# Qwen3-Embedding E2E Notebook — Review

**Readiness: Needs revision**  
**Review date:** 4 October 2026 (relay batch 2026-10-02)  
**Repository:** `kurtvalcorza/qwen3-embedding-pipeline`  
**Notebook:** `tutorials/qwen3_embedding_colab.ipynb`  
**Reviewed commit:** `f9f5c81a807762ea523d1c39e66613ef60cc003f` (origin/main)  
**Notebook Git blob:** `f3475b936f4b26bff23ab1da7208d3bae659d95a`  
**Finding prefix:** `QEM`

## Executive assessment

This is a well-built standalone E2E notebook. It pins the snapshot by digest and carries three modules verbatim. It uses a digest-pinned real corpus (Banking77, 616 / 154 / 385 pairs balanced over 77 intents) and reports retrieval metrics beside a random floor and a lexical baseline. Fine-tuning is a bounded InfoNCE pass over the last two decoder layers, with the epoch chosen on validation, and the safetensors adapter reloads with exact parity. The default path has a clean-runtime record of this exact blob (Kaggle T4, 2026-09-19). In this review, all 11 code cells also ran on a CPU workstation, and the CPU float32 numbers the notebook quotes reproduced exactly (frozen recall@1 0.6338 → adapted 0.8026, MRR 0.7422 → 0.8785).

It is not yet ready for its declared `GUIDED` audience:

- **Restart:** the default Run all needed one manual restart after the in-kernel install.
- **Re-runs:** a re-run of any section after the default run starts from the already-adapted model. That covers the optional experiments and the BYOD re-run from Section 4 that the opening prescribes. A re-run of Section 6 printed the adapted scores under `frozen_model_test`.
- **BYOD size:** BYOD cannot reach fine-tuning with fewer than 50 records, although the notebook states a minimum of 8. The error then names a count the user did not supply.
- **Guided layer:** it is mostly absent.

These four Major findings concern different journeys. None of them shows that a recorded default-path metric is wrong.

## 1. Review contract and evidence

| Item | Scope |
|---|---|
| Profile / mode | `E2E` / `GUIDED`. The notebook declares NOTEBOOK_SPEC `2.0`; the current fleet spec is `2.2` (2026-09-26) and is applied here. GDL1–GDL15 and EXE1–EXE7 are `SHOULD` requirements |
| Audience (stated) | Basic Python and NumPy, plus knowing dense vectors, cosine similarity, recall@k / MRR and InfoNCE. There is no explicit statement of the intended learner |
| Supported runtime (stated) | "Google Colab or Jupyter, Python 3.12". CPU float32; CUDA in bfloat16 when present |
| Model | `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d…` (Apache-2.0), 11-file manifest, 1,207,487,471 B |
| Default data | Banking77 (CC BY 4.0), two pinned CSVs (`b06e26ac…` / `d12d6e3b…`). Train and validation come from the release `train` member and test from the release `test` member, 8 / 2 / 5 per intent |
| Adaptation | Last 2 of 28 decoder layers (31,461,888 of 595,776,512 params); AdamW 5e-5, 2 epochs, batch 16, τ 0.05; texts truncated to 64 tokens in training; the epoch is chosen on validation MRR |
| Promised outcomes | Pinned install; snapshot staging and verification; corpus fetch, validation, a leakage-free split and 4 refusals; the inference contract with input manifest and batch refusal; floor, lexical and frozen metrics; bounded fine-tuning; held-out four-way comparison; top-3 lists before and after; adapter export and reload parity; BYOD through the same stages; optional experiments |
| Generating revision | `metadata.dimer.generated_from.revision` = `5e2f37d79d65`. That commit is not an ancestor of main, but `git diff 5e2f37d7 origin/main -- src` is empty, so the carried modules are byte-identical to main's |

### Evidence obtained

**Source inspection:** the whole notebook (25 cells) at the reviewed SHA; the three carried modules; `tools/build_notebook.py`; `tools/notebook_template.py`; `tutorials/README.md`; `docs/release-verification.md`; `STATUS.md`.

**Documented execution evidence:** `docs/release-verification.md` records a clean Kaggle Tesla T4 run of **this exact blob** (`ec6dd95` / `f3475b93`, 2026-09-19): PASSED, 11/11 code cells, **1 restart after the install cell**, 323.6 s. In the archived `run_summary.json`, pass 1 raised `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime…`, and `restarted_after_install_cell` is `true`. I read the archived `executed.ipynb` outputs and compared them with every number quoted in the prose. No Colab record exists for this notebook. The 2026-10-04 Colab record in that file belongs to the sibling workshop notebook.

**Direct execution (this review):** `run_probes.py` ran on a Windows workstation CPU with `CUDA_VISIBLE_DEVICES=-1`, Python 3.12.14, torch 2.13.0+cpu (the pin is 2.14.0), transformers 4.57.6 and NumPy 2.5.3. The install cell was skipped with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`. The pinned snapshot was hard-linked from a local checkout and the two Banking77 CSVs came from a local cache, with digests checked. The probe executed the notebook's own code cells in order. This is not Colab and not a clean runtime.
- `default`: 11/11 code cells ok in 418 s. Outputs: frozen test recall@1 0.6338 / MRR 0.7422; adapted 0.8026 / 0.8785; validation MRR 0.699 → 0.812 → 0.828; reload parity exact (MRR 0.859199 both ways); 6 exports.
- `al`: Sections 1–6 on the pristine base, then the default run's own exported adapter overlaid with `load_artifact` (an exact reproduction of the post-default state, since parity was exact), then a re-run of Section 6. A re-run of Section 7 with `TRAINABLE_LAYERS = 1` was started twice and stopped both times at the 590 s probe cap. It is **not verified**.
- `static`, `byod_min`: source probes, a cross-split near-duplicate scan of the drawn sample, and the carried Section 4 split and validation logic run on synthetic BYOD-shaped records (no model, no upload widget).

**Not verified:** a fresh Colab run; the upload widget; any BYOD run through the model; the `TRAINABLE_LAYERS = 1` re-run outcome; measured learner understanding.

## 2. Separate judgments

| Dimension | Judgment |
|---|---|
| Technical correctness | The default path is sound, recorded and reproduced on CPU. Snapshot, corpus and artifact handling are digest-checked and `adapt` is transactional. The defects are re-run state (no reset to pretrained), the BYOD size arithmetic, and BYOD asserts with no message. |
| Scientific validity | Splits use the release partition and are exact-disjoint. Floor and lexical baselines run on the same 385 queries and 77 documents, the test split is used only for final scoring, and "one seeded split, no dispersion" is stated. 9 / 385 test queries are near-duplicates of a training or validation message (QEM-m2). |
| Promise fulfilment | Default promises are delivered and match the record. BYOD is promised to pass "through the same … fine-tuning … reload-parity cells", but it fails below 50 records (QEM-M3). The optional experiments do not measure what they say (QEM-M2). The input manifest covers 64 of the "77 documents" (QEM-m3). |
| Learner experience | The prose is precise and candid about limits. It is dense reference prose with no how-to-use, roadmap, glossary, predictions, checkpoints, Infrastructure labels or troubleshooting (QEM-M4). Section 7 quotes CPU numbers without a device label (QEM-m4). |
| Spec conformance | Fails these `MUST`s: RUN10 / ENV6 (restart), DAT12 and DAT19 (stated minimum wrong, error not actionable), DAT14 (small BYOD cannot reach adaptation). `SHOULD` deviations: GDL1–GDL14, EXE2, EXE5, SPL10, and the declared spec 2.0. |

### Promise → evidence trace (summary)

| Claim | Cell | Observable result | Status |
|---|---|---|---|
| Pinned install, Run all with no configuration edit | 3 | Kaggle: pass 1 RuntimeError, restart, pass 2 ok | Fails RUN10 (QEM-M1) |
| Snapshot staged and verified, 11 files | 11 | Kaggle: 26 files / 1209 MB staged; CPU: 11 verified | Delivered |
| 10,003 + 3,080 rows, 616/154/385, 77 docs, 4 refusals | 13 | CPU: as claimed, digests `dacba395…`/`3d07cfd5…`/`df00c83e…`, 4/4 rejected | Delivered |
| "without leakage" | 13 | 0 exact (normalised) cross-split repeats; 9/385 near-duplicates | Partly (QEM-m2) |
| Input manifest "for the 77 documents" | 15 | `validate_inputs(document_set[:MAX_BATCH])` = 64 documents | Partly (QEM-m3) |
| Frozen well above floor and lexical ("sixties / seventies") | 17 | Kaggle 0.631 / 0.741; CPU 0.634 / 0.742 | Delivered; prose matches |
| Val recall@1 "+ten points or so", recall@5 "passes 95 %" | 19 | Kaggle 57.8 → 67.5 %, r@5 95.5 %; CPU 58.4 → 72.7 %, r@5 97.4 % | Delivered |
| "two layers at 5e-5 80.3 % … the default" | 18 | CPU 80.26 %; Kaggle T4 (CUDA bf16) 76.6 % | CPU-only figure (QEM-m4) |
| Held-out recall@1 +≥10 pts, MRR +≈0.1 | 21 | Kaggle +13.5 pts / +0.109; CPU +16.9 / +0.136 | Delivered |
| Adapter reload parity | 23 | Kaggle 0.812533 both ways; CPU 0.859199 both ways | Delivered |
| BYOD through the same stages | 13→23 | Fewer than 50 records: ValueError at Section 4 | Fails DAT14 (QEM-M3) |
| Optional experiments (layers, τ, epochs, instruction) | 17/19 rerun | Section 6 re-run scored the adapted model as "frozen" (probe `al`) | Misleading (QEM-M2) |

### Objective → activity trace (summary)

The opening lists 10 objectives, phrased as things the notebook does ("install…, read…, stage…, fetch…, embed…, read recall@k…, run…, evaluate…, compare…, export…"). Each has executing code. The learner activity is running cells and reading printed dicts. Two objectives ask for interpretation: "read recall@k and MRR … understand what they do and do not measure" and "compare retrieved documents before and after". Neither has a prompt, a prediction or a checkpoint that would show the learner did so. The only change-a-parameter activity is *Optional experiments*, and it gives invalid comparisons without a reset (QEM-M2, QEM-M4).

## 3. Findings

### Major

#### QEM-M1 — Run all needs a manual restart after the in-kernel install

**Cell/section:** Section 1, cell 3 (generated by `tools/build_notebook.py`, install block lines ~55–70).  
**Observed issue:** The cell runs `pip install` for torch 2.14.0, torchvision, torchaudio, transformers 4.57.6, huggingface-hub, safetensors and numpy 2.5.3 into the running kernel. When a distribution that was already loaded changes, it raises `RuntimeError(... Restart the runtime, then rerun from the top.)`.  
**Consequence:** Colab and Kaggle both preload NumPy and torch. On a hosted image whose preloaded versions differ from the pins, the first Run all stops at cell 3, and the learner must restart and run again. RUN10 and ENV6 forbid this.  
**Evidence:** Documented execution, exact blob: Kaggle T4 `run_summary.json`, pass 1 `RuntimeError … cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`, `restarted_after_install_cell: true`. `docs/release-verification.md` itself records "(1 restart after install cell)". Source: probe `static` (`install_cell_pip_into_kernel: true`, `uses_uv: false`).  
**Recommended correction:** Replace the in-kernel install with the fleet's **uv isolated-environment pattern**:
1. A carrier cell bootstraps uv.
2. It creates a managed-Python venv (`uv venv --managed-python --python 3.12.12 <ROOT>/env`).
3. It installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`.
4. The workload runs in that env, so the kernel's preloaded NumPy and torch are never replaced.

Reference: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on origin/main. The same repository already ships this pattern for its workshop notebook (`tools/semantic_search_isolated_runtime.py` and `tutorials/requirements-semantic-search-workshop.lock.txt`; Colab T4 run of blob `d72ea820` on 2026-10-04 with no restart). Make the change in `tools/build_notebook.py` / `tools/notebook_template.py`, not by hand.  
**Acceptance check:** In a fresh Colab and a fresh Kaggle runtime, Run all completes in one pass with zero restarts, and the run record says "0 restarts".  
**Spec:** RUN10, ENV6 (MUST); REL11.

#### QEM-M2 — Re-runs continue from the already-adapted model, so "frozen" and epoch 0 are not frozen

**Cell/section:**
- The opening ("set `USE_BYOD = True` in Section 4 and re-run from that cell").
- Interpretation, *Optional experiments*: "set `TRAINABLE_LAYERS = 1` and watch the gain shrink … `TEMPERATURE = 0.2` … `EPOCHS = 4` … change `INSTRUCTION` and re-read the frozen numbers".
- Sections 6–9 (cells 17, 19, 21, 23).

`pipe` is built once in Section 3. `Qwen3EmbeddingPipeline.adapt` (`pipeline.py` ~410–562) starts from the current weights, and its epoch-0 entry is always written with `"note": "frozen model"`.  
**Observed issue:** No cell restores the pretrained weights, and no text says which cells to re-run. After the default run:
- re-running Section 6 scores the adapted model as `frozen_model_test`;
- re-running Section 7 with a changed form value trains the adapted model again, while epoch 0 is still labelled `frozen model`. With `TRAINABLE_LAYERS = 1`, the new run trains only layer 27, and layer 26 keeps its update from the first run. `save_artifact` then exports only the tensors in `trainable_names` (layer 27), so the artifact no longer describes the model that was evaluated. The Section 9 parity assertion compares the in-memory model against `from_artifact`, which is the pristine base plus layer 27. It should fail with a bare `AssertionError` (inferred);
- the BYOD re-run from Section 4 makes the Section 5 top-3 lists, the Section 6 "frozen" baseline and epoch 0 all the Banking77-adapted model.

**Consequence:** The optional experiments compare accumulated training against a mislabelled baseline. "Watch the gain shrink" with one layer cannot be read, because the starting point already carries the two-layer gain. BYOD learners see a "frozen" number that is not the pretrained model. These are the conclusions the notebook explicitly teaches learners to draw.  
**Evidence:** Direct execution (probe `al`, CPU, real model). On the pristine base, Section 6 gave frozen recall@1 0.6338 / MRR 0.7422 with `adapted: False`. After the default run's adapter was overlaid, re-running Section 6 printed `frozen_model_test` recall@1 **0.8026** / MRR **0.8785**, identical to the adapted score, with `adapted: True`. Source: `adapt` has no reset and the literal `"note": "frozen model"` is in it (probe `static`); `save_artifact` writes only `self.adapter["trainable_names"]` (`pipeline.py` ~566–590). **Not verified:** the `TRAINABLE_LAYERS = 1` re-run itself, which was stopped at the probe-time cap.  
**Recommended correction:** In `tools/notebook_template.py`, rebuild the pipeline from the verified snapshot at the top of Section 6 (`pipe = Qwen3EmbeddingPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)`) and again before `pipe.adapt` in Section 7. Alternatively, have `adapt` refuse (or reset) when `self.adapter is not None`, with an actionable message. State in the opening and in *Optional experiments* exactly which cells to re-run.  
**Acceptance check:**
1. After a default run, re-running Section 6 reproduces the first-run frozen metrics with `adapted: False`.
2. Re-running Section 7 with `TRAINABLE_LAYERS = 1` logs an epoch-0 validation MRR equal to the first run's epoch 0 (0.699 on CPU), and Section 9's parity assertion passes.
3. A BYOD re-run reports a frozen baseline equal to a fresh pretrained model on the same split.

**Spec:** SRC2 (hidden state dependency), GDL10, UX7; ART8 / OUT8 (the exported variant must be the evaluated one).

#### QEM-M3 — BYOD below 50 records cannot reach fine-tuning, though the stated minimum is 8, and the error names the wrong count

**Cell/section:** Section 4, cell 13 (template ~144–175): `splits = split_dataset(records, seed=SPLIT_SEED)` and then `dataset_manifests = {name: validate_dataset(part) …}`. `validate_dataset` defaults to `min_records=8` for **each split**, including validation and test. Also affected: Section 6 cell 17 and Section 8 cell 21 asserts (template lines 266, 345); Prerequisites "a dataset needs 8..20,000 records".  
**Observed issue:** `split_dataset` gives 20 % to test and 15 % to validation. The validation split reaches 8 records only when the dataset has 50 or more. Smaller datasets fail at Section 4:
- 8 records fail inside `split_dataset` ("split leaves 5 training records").
- 12–49 records fail with `ValueError: <n> records; 8..20000 are required`, where `<n>` is the size of the validation or test split. The user, who supplied 12–49 records against a stated minimum of 8, is shown a count they never gave and a range they already satisfy.

Two further asserts run in BYOD mode and have no message: `frozen_test['mrr'] > floor['mrr']` and `adapted_test['mrr'] > frozen_test['mrr']`. A small or hard BYOD set that is not improved by two epochs stops at Section 8 with a bare `AssertionError`, before export and reload.  
**Consequence:** A learner trying BYOD with a small first file, the natural first try, cannot reach adaptation. The error does not identify the failed contract.  
**Evidence:** Direct execution of the carried Section 4 logic (probes `byod_arithmetic`, `byod_min`: synthetic records over 8 positives, `seed=42`):

| Records | Outcome |
|---|---|
| 8 | split error |
| 12 | `2 records; 8..20000 are required` |
| 20 | `4 records; …` |
| 38 | `6 records; …` |
| 50 | **smallest passing** |
| 50–120 | all pass |

The asserts are from source inspection; that a real BYOD set fails them is inferred, not executed. The upload widget was not exercised.  
**Recommended correction:**
- In `samples.py` / `notebook_template.py`, validate the validation and test splits with `min_records=1` (as `evaluate` and `adapt` already do), or compute and state the true minimum.
- Have `split_dataset` raise a message that names the split, its size and the dataset size required.
- Make the Prerequisites and the Section 4 text state the effective minimum.
- Guard the two metric asserts with `if not USE_BYOD:`. In BYOD mode, print an interpretation line and continue.

**Acceptance check:**
1. A 20-record BYOD CSV over at least 2 positives reaches Section 9 (adapt, evaluate, export, reload parity) without editing cells.
2. A dataset below the true minimum is rejected with a message naming the dataset size and the minimum.
3. The sample path still asserts.

**Spec:** DAT12, DAT19, DAT14 (MUST); DAT13.

#### QEM-M4 — Declared GUIDED, but the guided layer is largely missing

**Cell/section:** Whole notebook. The headings are only Prerequisites, Sections 1–9, Interpretation and References (probe `static`).  
**Observed issue:** The following are missing:
- an intended-learner statement (GDL1);
- **How to use this notebook** (GDL2);
- a roadmap (GDL3);
- a glossary for InfoNCE, MRR, last-token pooling and instruction-aware queries (GDL5 / GDL6);
- question-and-predict prompts before Sections 6, 7 and 8 (GDL7);
- interpretation checkpoints with sample answers (GDL9);
- a Predict → Change → Run → Observe → Explain activity (GDL10);
- **Infrastructure** labels on the 34 kB pipeline cell, the 15 kB samples cell, the 3 kB manifest cell and the install cell (GDL11);
- a core / evaluation / engineering separation (GDL12);
- troubleshooting for download failure, a digest mismatch, a restart, out-of-memory and BYOD (GDL13);
- a conclusion template (GDL14).

The Section 4–9 introductions do give "Look for" notes (GDL8 partly met).  
**Consequence:** The stated audience ("basic Python and NumPy") meets three verbatim module cells (about 54 kB of code) before any model runs. They get no signal that those cells are infrastructure, and no point at which they are asked to interpret anything. The notebook reads as an expert reference pipeline, not the guided tutorial it declares itself to be.  
**Evidence:** Source inspection (probe `static`: 9 of 10 guided markers absent; the single "checkpoint" hit is the phrase "1.19 GB checkpoint download").  
**Recommended correction:** In `tools/notebook_template.py`, add:
- a How-to-use / roadmap cell after the opening;
- Infrastructure callouts on cells 3, 5, 7, 9 and 11;
- a prediction prompt before Sections 6, 7 and 8;
- one checkpoint with a collapsible worked answer after Section 8, for example "why is the lexical baseline at 31 % when the documents are the intent names?";
- a troubleshooting section;
- a conclusion template in the Interpretation section.

The sibling workshop in this repository (`DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb`, GDL7–12 added 2026-09-26) is a local model.  
**Acceptance check:** Each of GDL1–GDL14 maps to a named cell, and a reviewer can point to one Predict → Change → Run → Observe → Explain activity that does not alter the canonical run.  
**Spec:** GDL1–GDL14 (SHOULD; Major for the declared `GUIDED` audience).

### Minor

#### QEM-m1 — BYOD imports `google.colab` with no path field, though Jupyter is a stated supported runtime

**Cell/section:** Section 4, cell 13 (template line 149).  
**Observed issue:** `if USE_BYOD: from google.colab import files; uploaded = files.upload()`. There is no `BYOD_PATH` field.  
**Consequence:** On Jupyter, which the Prerequisites list as a supported runtime, BYOD fails with `ModuleNotFoundError`. `next(iter(uploaded.items()))` also raises a bare `StopIteration` when the upload dialog is cancelled.  
**Evidence:** Source inspection (probe `static`: `byod_imports_google_colab: true`, `byod_path_param: false`).  
**Recommended correction:** Add `BYOD_PATH = ''  # @param`. When it is set, read that file and do not import `google.colab`. Otherwise, upload, and reject an empty upload with a message.  
**Acceptance check:** On a non-Colab Jupyter kernel, `USE_BYOD = True` with `BYOD_PATH` set to a valid CSV completes Section 4. A cancelled upload prints an actionable message.  
**Spec:** EXE2 (SHOULD), DAT16, DAT19.

#### QEM-m2 — "Without leakage" is checked only as case-insensitive exact repeats; 9 of 385 test queries are near-duplicates of a training or validation message

**Cell/section:** Section 4 prose ("validated and split without leakage"); `check_split_disjoint` (`samples.py` ~269) compares `query.lower()` only.  
**Observed issue:** On the drawn default sample, no test query repeats a training or validation message after punctuation and space normalisation. But 9 test queries have a token Jaccard ≥ 0.8 with a training or validation message, and 7 of the 8 shown share the intent. Examples: "My top-up has been cancelled." / "Has my top-up been cancelled?" (Jaccard 1.0); "I got my American Express in Apple Pay, why is top up not working?" / "… not working correctly?" (0.93). At release level, 25 test rows equal a training row after normalisation.  
**Consequence:** The effect on reported metrics is small (≤ 2.3 % of test queries). But the notebook teaches de-duplication as the leakage control, and its own check would miss these near-duplicates in user data.  
**Evidence:** Direct execution (probe `static` → `leakage`, on the carried `build_sample_dataset(seed=42)` output).  
**Recommended correction:** Normalise punctuation and whitespace in `check_split_disjoint`. Report (not necessarily refuse) high-overlap cross-split pairs. Reword the claim to "no exact repeats across splits; near-paraphrases exist in Banking77".  
**Acceptance check:** Section 4 prints a near-duplicate count for the sample (9 at Jaccard ≥ 0.8), and the prose no longer claims leakage-freedom beyond what is checked.  
**Spec:** SPL10 (SHOULD), SPL3.

#### QEM-m3 — The input manifest covers 64 of the 77 documents it claims to cover

**Cell/section:** Section 5, cell 15 (template line 206): `validate_inputs(document_set[:MAX_BATCH], …)`. The prose says "it returns an input manifest for the 77 documents".  
**Observed issue:** `MAX_BATCH` is 64, so 13 documents are absent from `outputs/qwen3_embedding_input_manifest.json` and from `result.json`'s `inference_contract`. They are still embedded, and `embed` validates them, so nothing unsafe runs.  
**Consequence:** The exported manifest does not describe the inputs it claims to. A BYOD document set larger than 64 is likewise partly recorded.  
**Evidence:** Source inspection (probe `static`: `input_manifest_slice: true`, `prose_claims_77_manifest: true`).  
**Recommended correction:** Validate in `MAX_BATCH` chunks and merge the manifests, or say "the first 64 documents (one batch)".  
**Acceptance check:** The exported input manifest lists 77 document entries on the sample path, or the prose matches what is recorded.  
**Spec:** INF3 / promise fulfilment.

#### QEM-m4 — Section 7 quotes CPU float32 figures as "the default" without a device label

**Cell/section:** Section 7 markdown (cell 18): "two layers at 5e-5 80.3 % with a 126 MB adapter — the default".  
**Observed issue:** 80.3 % is the CPU float32 build-record figure. The only hosted record, Kaggle T4 in CUDA bfloat16 on this exact blob, gave **76.6 %** adapted recall@1, validation MRR 0.788 (CPU 0.828) and document self-cosine median 0.846 (CPU 0.680). The Prerequisites mention a precision difference in general, but this sentence names a specific number as "the default".  
**Consequence:** A Colab GPU learner comparing against the quoted default sees a 3.7-point shortfall and may suspect a broken run.  
**Evidence:** Documented execution (archived Kaggle `executed.ipynb`) versus direct execution (CPU probe `default`, 80.26 %).  
**Recommended correction:** Label the sweep "CPU float32 build record" and give the hosted CUDA figure or a range. Alternatively, phrase expected results qualitatively, as Sections 6 and 8 already do.  
**Acceptance check:** Every number quoted in the markdown names its device and precision, or is a range that contains both the CPU and the T4 records.  
**Spec:** GDL8 (no hard-coded result that varies legitimately).

### Suggestions

- **QEM-S1** — Regenerate against NOTEBOOK_SPEC 2.2. The notebook declares 2.0 in its metadata, opening and references.
- **QEM-S2** — Record a Colab run of the primary tutorial. Colab is the "supported user path" in `docs/release-verification.md`, but this notebook has only a Kaggle T4 record.
- **QEM-S3** — Document `DIMER_NOTEBOOK_CI_PREINSTALLED` in the notebook. Cell 3 reads it, but no markdown mentions it (EXE5).
- **QEM-S4** — The recorded generating revision `5e2f37d7` is outside main's history. Its `src/` is identical to main's, so regenerate to record a reachable revision.
- **QEM-S5** — Report the lexical baseline's tie count. With pessimistic ties and Jaccard on two- to three-word documents, many queries tie at 0. The gap to the embedder partly reflects tie policy, as the sibling workshop's QSR-01 fix made visible (93/154 tied).
- **QEM-S6** — `gold_rank_frozen` / `gold_rank_adapted` in `retrieval.csv` are ranks within the top 3 only (`None` beyond). Name them `gold_rank_in_top3`, or compute the full rank.

### Checked and not raised

- **Test/train leakage (exact):** splits come from the release's own train/test members. Training and validation are cursor-disjoint, and 0 normalised exact repeats cross splits (probe `leakage`). The near-duplicate residue is QEM-m2.
- **Stale worked answers:** the Section 6 ("sixties / seventies"), Section 7 ("+ten points or so", "recall@5 passes 95 %"), Section 8 ("+ten points or more, MRR about a tenth") and Interpretation statements all hold on both the Kaggle T4 record and the CPU probe. The only device-specific figure is QEM-m4.
- **Score semantics:** "cosine is a similarity, not a probability", no threshold, and adapted cosines not comparable to frozen ones are stated correctly (UNC1–UNC4).
- **Artifact:** the safetensors adapter, the manifest with base revision and digest, and pre-deserialisation checks are present. Reload parity was exact on CPU and on T4 (ART1–ART5, VER2, VER4).

## 4. Readiness

**Needs revision.** Four Major findings are open. Applicable `MUST`s fail: RUN10/ENV6 (QEM-M1), DAT12 / DAT19 / DAT14 (QEM-M3). Exact-blob execution evidence exists for the default path only, on Kaggle and with one restart.

Remaining gates after fixes:
1. A one-pass fresh-runtime record (Colab preferred) of the regenerated blob, with 0 restarts.
2. The re-run checks in QEM-M2's acceptance check, including the `TRAINABLE_LAYERS = 1` parity.
3. A real BYOD run of 20 records through reload parity, plus one actionable rejection (REL12).

## 5. Verified versus inferred

- **Verified by direct execution (CPU, local env, notebook cells):** the default path (11/11, CPU metrics reproduce the quoted build record); the re-run of Section 6 scoring the adapted model as "frozen"; the BYOD size threshold of 50 and its error text; the near-duplicate scan.
- **Verified by documented execution:** the default-path outcome and the restart, from the archived Kaggle record of this exact blob.
- **Inferred:**
  - that a `TRAINABLE_LAYERS = 1` re-run fails the Section 9 parity assertion (from `adapt` / `save_artifact` source; the direct run hit the probe cap);
  - that Colab behaves as Kaggle does at the install cell;
  - that the Section 8 assert stops some real BYOD sets.
- **Not verified:** the upload widget; a full BYOD model run; learner understanding.

**Finding most likely to be wrong:** the inferred part of QEM-M2. If `adapt` on the second run selects epoch 0, it restores `best_state` captured at epoch 0, which covers only layer 27. Layer 26 still differs from the base, so I expect parity to fail either way. But this was not executed, and the parity sub-claim rests on source reading alone. The "frozen" mislabel in the same finding is verified.
