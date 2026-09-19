# Weight provenance and DIMER hosting

- Upstream: `Qwen/Qwen3-Embedding-0.6B`
- Immutable revision: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- Weight format: SafeTensors (`model.safetensors`, 1,191,586,416 bytes, bfloat16)
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README)
- Local layout: `weights/qwen3-embedding-0.6b/` holds the 11 files listed in `dimer-base-manifest.json` (`config.json`, `1_Pooling/config.json`, tokenizer files, `model.safetensors`, upstream `README.md`, sentence-transformers configs) with byte size and SHA-256 for each. `verify_snapshot()` in `src/qwen3_embedding_pipeline/pipeline.py` checks all of them before any load; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision into that directory. `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.
- DIMER hosting: Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the license and notices. DIMER may mirror the pinned checkpoint in its model store under those terms.
- Loader trust boundary: Transformers `AutoModel` (`Qwen3ForCausalLM` config, used as an encoder) + `AutoTokenizer` with `trust_remote_code=False`; the loader reads only the verified local directory (`local_files_only=True`) and falls back to the Hub at the pinned revision only when `allow_download=True` is passed explicitly.

## Adaptation corpus (tutorial data, not weights)

- Corpus: Banking77 (Casanueva et al., NLP4ConvAI 2020), `PolyAI-LDN/task-specific-datasets` at commit `57ec275d8078af65b7731c2a98be812d844a6d6b`, licence CC BY 4.0.
- Files: `banking_data/train.csv` (839,073 bytes, SHA-256 `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`) and `banking_data/test.csv` (239,961 bytes, `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`), fetched by `samples.fetch_corpus` from `raw.githubusercontent.com` over HTTPS at run time into the git-ignored `weights/banking77/` cache and refused on any byte or SHA-256 mismatch.
- Sample: `build_sample_dataset(seed=42)` draws 8 / 2 training / validation pairs per intent from `train.csv` (disjoint messages) and 5 test pairs per intent from `test.csv` (616 / 154 / 385); every intent name becomes the pair's positive document (`card_arrival` → `card arrival`). The repository redistributes none of the corpus; DIMER hosting of the weights is unaffected.
- Adapter artifacts written by the tutorial (`outputs/qwen3_embedding_adapter/`, `org.valcorza.qwen3-embedding-0.6b.adapter.v1`, about 126 MB) carry only the trained decoder-layer tensors and a manifest naming the base `model.safetensors` digest and the training instruction; they are outputs, not hosted weights.
