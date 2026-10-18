# Pre-registration amendment 1 (2026-10-18): float32 training precision and fresh test 2

Amends `docs/qwen3-embedding-preregistration.md` (2026-10-17, commit `3b4844a`). This file is committed **alone and
before any code change** of this amendment, and before any run of the amended recipe. Owner decision (Kurt,
2026-10-18): keep the current record, fix the GPU training precision, and pre-register a second fresh test.

## 0. Why

The hosted T4 run of commit `1cc43ed` (blob `971187e6`, 2026-10-17) is the pre-registered evaluation of record for
recipe 1. On the fresh test, MRR went 0.7338 → 0.8455, a paired difference of +0.112 with a query-level 95 % CI of
[0.091, 0.134] (`docs/release-verification.md`). Since then the orchestrator has verified one finding:

- On CUDA the pipeline loads the model in bfloat16 (`pipeline.py:298`).
- AdamW then steps directly on those bfloat16 parameters (`pipeline.py:513`), with no float32 master copy. Learning-rate-sized updates (2e-5 to 5e-5) are partly lost to bfloat16 rounding.
- The result: validation MRR for C0 was 0.788 on the T4 against 0.828 on CPU float32, and C1 and C2 lost more (0.720 / 0.730 against 0.796 / 0.827).

Recipe 1 stays as recorded. Its fresh test has now been scored, so that test is exposed.

## 1. (a) Corrected precision recipe (recipe 2)

**Full float32 on every device.** The model is loaded in float32 on CUDA as on CPU. The trainable decoder layers are held and updated in float32, AdamW's moment estimates are float32, and the forward pass runs in float32 with no autocast. CPU was already float32 and is unchanged.

**Why not the alternatives.**

| Option | Why it was not chosen |
|---|---|
| bf16 autocast with float32 trainable layers | A T4 (sm_75) has no bfloat16 tensor cores. bf16 matmuls run without tensor-core acceleration, so autocasting to bf16 buys no speed over float32 on this GPU and adds a cast path. |
| fp16 autocast + GradScaler | fp16 is the only fast format on a T4. But Qwen3 was trained in bf16, and fp16 activations of a bf16-trained decoder can overflow in the frozen layers. That would make a silent numerical fault more likely than a speed problem, and it adds a scaler with its own skip-step behaviour. |
| Frozen layers in half precision, trainable layers in float32 | The forward pass mixes dtypes at the layer boundary, so it needs autocast. That leaves the same choice between the two rows above. |

**Memory on a 16 GB T4.**

| Item | Size |
|---|---|
| Float32 weights (595.8 M parameters) | about 2.4 GB |
| Gradients + AdamW moments for C2 (4 layers, 62.9 M parameters) | about 0.75 GB |
| Gradients + AdamW moments for C0 / C1 (2 layers, 31.5 M parameters) | about 0.38 GB |
| Activations (batch 16, 64 tokens) | small |
| Evaluation (batches of 64 short texts) | small |
| **Expected peak** | **below 5 GB** |

**Time (estimate, not a measurement).** The recipe-1 T4 times were 138.9 s for the grid, 47.7 s for the selected retrain, 11.1 s for the transfer check and 382.9 s for the whole session, all in bf16. Because bf16 had no tensor-core path on this GPU, float32 is expected to cost roughly 1–2× those model times:

- grid: about 2.5–5 min;
- whole session: about 7–12 min;

All of these are well below the 55-minute bound that keeps the grid on by default.

**Adapter size.** The adapter is saved in float32: 125,849,896 bytes (about 126 MB) for 2 layers, and about 252 MB for C2. A recipe-1 T4 adapter was bf16: 62,926,128 bytes (about 63 MB). The manifest records the dtype.

## 2. (b) Fresh test 2

- **Pool.** Banking77 test.csv after `filter_records` (3,079), minus every message any run has scored or excluded. That removes 516 + 22 + 770 = 1,308 and leaves **1,770 never-scored messages**, 19–24 per intent. All 77 intents reach 10.
  - The tutorial's exposed development test: 385.
  - The workshop W16's 154 (23 shared with the 385; union 516).
  - The 22 exact `query_key` repeats of a train.csv message.
  - Fresh test 1 (770, manifest `fe68b43a…`).
- **Draw.** `random.Random(20261018)`. For each of the 77 intents in sorted order, the clean candidates are sorted by source id and shuffled, and the first 10 are taken. The resulting **770** are then shuffled. Implemented as `samples.build_fresh_test_2`.
- **Pin.** The manifest SHA-256 of the sorted (source id, query) pairs is
  `cd0171285dfd6ccc72b88d82b43ad856af6869a84a49cc50ac06046119f37a63`
  (`samples.FRESH_TEST_2_MANIFEST_SHA256`). The builder refuses any other draw and asserts that the draw is disjoint from the exposed 385, the W16 154, fresh test 1 and every train.csv message.
- **Near-duplicates** (token Jaccard ≥ 0.8 against the training and validation draw) are reported, not removed: 12 in the draw and 29 in the clean pool.

## 3. (c) Unchanged

These are all unchanged:

- the data source, split seed and sizes;
- the instruction;
- the epochs, batch size, temperature, token budget, adapt seed and optimiser;
- epoch selection on validation MRR;
- the grid (C0 / C1 / C2), the 0.02 margins and the keep-frozen rule;
- the paired bootstrap, with the query-level CI as the pre-registered interval and the intent-level CI as the disclosed secondary analysis (2,000 resamples, seed 0);
- the display-only CLINC150 transfer check with its two domain rows.

## 4. (d) Fresh test 1

From this amendment on, fresh test 1 is labelled **"fresh test 1 (scored once, 2026-10-17, bf16 recipe; now an exposed test)"**. It is reported for continuity, like the exposed development test, and is never used for any selection or stop.

## 5. Score-once wiring for recipe 2

- `preregistered` is computed over all the settings above plus `precision = float32`. The fresh test 2 manifest replaces fresh test 1's in the pre-registered settings.
- **Only fresh test 2 is scored as the pre-registered evaluation.** Every other setting is exploratory, as in the original pre-registration.
- `samples.RECORDED_EVALUATIONS` keeps the 2026-10-17 run as the recipe-1 record. The recipe-2 entry stays `None` until the first hosted pre-registered T4 run of recipe 2 is recorded. That run is THE evaluation of record for recipe 2, and later identical hosted runs are reproductions.
- CPU development checks by the fixer run with **both** fresh tests masked.
