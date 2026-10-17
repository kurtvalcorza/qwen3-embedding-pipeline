# Pre-registration: validation re-selection and fresh test for `qwen3_embedding_colab` (2026-10-17)

Written and committed **before** any run of the grid below and before any scoring of the fresh test. It answers
review finding R72-M1 (row 72 of the 2026-10-15 tutorial triage): the default learning rate (5e-5 over 2e-5) and
trainable scope (2 over 4 layers) were compared on the 385 test queries in 2026-09 (PR #3 build record at
`9bcb449`), so those queries are no longer an independent test.

## 1. Fixed settings

| Setting | Value |
|---|---|
| Data source | Banking77, `PolyAI-LDN/task-specific-datasets` @ `57ec275d8078af65b7731c2a98be812d844a6d6b` (train.csv sha256 `b06e26ac…`, test.csv sha256 `d12d6e3b…`) |
| Split seed | `SPLIT_SEED = 42` (`samples.build_sample_dataset`) |
| Sizes | train 616, validation 154 (from train.csv); exposed development test 385 (seed-42 draw from test.csv); fresh test 770 |
| Instruction | `Given a customer support message, retrieve the banking intent it expresses` |
| Training | EPOCHS = 2, BATCH_SIZE = 16, TEMPERATURE = 0.05, MAX_TRAIN_TOKENS = 64, adapt seed 0, AdamW (weight decay 0.01), gradient clipping 1.0, no scheduler |
| Epoch selection | the epoch with the highest validation MRR (epoch 0 = frozen) |
| Documents | the 77 intent phrases |

## 2. Grid (validation only)

| Config | Trainable layers | Learning rate |
|---|---|---|
| C0 (current default) | 2 | 5e-5 |
| C1 | 2 | 2e-5 |
| C2 | 4 | 2e-5 |

**Selection metric:** best-epoch validation MRR over the 154 validation queries (77 documents).

**Rule (a predeclared practical tolerance, not a significance threshold):**
1. C0 is kept unless a challenger beats C0's validation MRR by at least **0.02**.
2. C2 (twice the adapter size) must also beat the best 2-layer configuration (max of C0, C1) by at least 0.02.
3. If both C1 and C2 qualify, C2 is selected only under rule 2; otherwise C1.
4. Ties and differences inside the margin keep the smaller/current configuration (C0).
5. **Keep frozen** unless the selected configuration beats the frozen validation MRR (epoch 0) by at least 0.02.
   If the frozen model is kept, the fresh test is scored for the frozen model only.

Implemented as `samples.select_configuration`. The grid is **on by default** in the notebook (`RUN_GRID = True`;
estimated about 3–4 min on a T4, 10–15 min on a workstation CPU). A grid run is selection-only: it records
validation and never reads the fresh test.

## 3. Fresh test

- **Pool:** test.csv after `filter_records` (3,079 messages) minus every message this repository has scored:
  the tutorial's 385 seed-42 test queries and the merged workshop W16's 154 (2 per intent, seed 42; 23 shared),
  516 in all, leaving 2,563 never scored.
- **Exclusions:** every exact repeat of any train.csv message under the repository's leakage key
  (`samples.query_key`: lower-case, punctuation and whitespace runs collapsed) is dropped: **22** messages. The
  review's normalisation counted 7; all 7 are among the 22, and the other 15 differ from a train.csv message only by
  punctuation such as `top-up` / `top up`.
- **Near-duplicates:** token Jaccard ≥ 0.8 against the training and validation draw are **reported, not removed**:
  39 in the clean pool, 10 in the drawn fresh test.
- **Draw:** `random.Random(20261017)`; for each of the 77 intents in sorted order, the clean candidates sorted by
  source id are shuffled and the first 10 taken; the 770 are then shuffled. Implemented as `samples.build_fresh_test`.
- **Pin:** manifest SHA-256 of the sorted (source id, query) pairs =
  `fe68b43ae56b9092ab92f6eeb610b2179222d1dfa04c3eebd9c658ac69c0908f` (`samples.FRESH_TEST_MANIFEST_SHA256`); the
  builder refuses any other draw and asserts disjointness from the exposed and training messages.
- **The old 385** are the **exposed development test**: shown in every run, labelled, for continuity.

## 4. Score-once wiring (SCORE_ONCE_RULE.md, 2026-10-17)

- `preregistered` is computed from **all** settings above (`samples.preregistration_diff`, which also requires the
  grid to have run in the session and the configuration to equal its selection).
- The fresh test is scored **only** when `preregistered` is true. Any other setting (a changed form field, BYOD, a
  different split seed or instruction, the Activity, the More experiments, a smaller batch for OOM) is exploratory:
  validation and the exposed development test only, written to separate `*_exploratory_<n>.json` files.
- Outputs carry `test_look` ("pre-registered evaluation (this runtime)" | "reproduction" | "not scored
  (exploratory)") and `settings_differing`.
- **CPU development looks:** none of the fresh test. The fixer's CPU checks run with the fresh test masked
  (validation and the exposed development test only). The first hosted T4 run with these settings is the
  evaluation of record; later identical hosted runs are reproductions (`samples.RECORDED_EVALUATION` stays `None`
  until that run is recorded in docs/release-verification.md).
- Gates move off the test: the frozen-above-floor and adapted-above-frozen stops read **validation**; outcomes on
  the fresh test are verdicts only. Reload parity is checked on the validation queries.
- Paired comparison on the fresh test: per-query reciprocal-rank difference (adapted − frozen), paired bootstrap 95 %
  CI (2,000 resamples, seed 0).

## 5. Transfer check (optional, display only)

Owner ruling on R72-M2: keep Banking77 with a DAT9 disclosure and add an optional transfer check on a corpus that
is not in MTEB.

- **Corpus:** CLINC150 (Larson et al., 2019), `clinc/oos-eval` @ `828f8093932c8fe6ca7936c3d2e52903b1c523de`,
  `data/data_full.json` (2,495,390 bytes, sha256 `36923c37…`), CC BY 3.0.
- **Not in MTEB:** no task, descriptive-stats file or dataset name containing `clinc` or `oos` in the MTEB tree at
  `embeddings-benchmark/mteb@a33f356042362f59b72af1415e176c19f9b2c771` (4,757 paths; the only code-search hit is
  an author name, "Clinchant"); the `mteb` Hugging Face organisation lists no CLINC dataset; the Qwen3-Embedding
  `evaluation/task_prompts.json` (sha256 `c9493103…`) has no CLINC entry. Whether CLINC150 was in Qwen3-Embedding's
  training data is undisclosed, as for Banking77.
- **Set:** 2 in-scope test queries per intent (300) against the 150 intent phrases, `random.Random(20261017)`;
  dataset digest `2f45326ae57c93c79a65f090be7a7623507415298613a15eb6553f960cfb9577`
  (`samples.TRANSFER_MANIFEST_SHA256`). Instruction: `Given a request to a virtual assistant, retrieve the intent it
  expresses` (the same for both models).
- **Comparison:** frozen vs the selected adapted model, MRR and recall@1, paired bootstrap 95 % CI on the MRR
  difference. Display only: never a selection input and never a gate. On by default (`RUN_TRANSFER_CHECK = True`;
  about 1 min on a T4, under the 5-minute bound).
