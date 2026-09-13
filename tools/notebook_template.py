"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "qwen3_embedding_pipeline",
    "repo_name": "qwen3-embedding-pipeline",
    "stem": "qwen3_embedding",
    "notebook_name": "qwen3_embedding_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "Qwen3EmbeddingPipeline",
    "weights_key": "qwen3-embedding-0.6b",
    "runtime_imports": ["torch", "transformers"],
    "title": "Qwen3-Embedding-0.6B — DIMER text embedding tutorial (standalone)",
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
    "capability": "text embeddings (1024-d, last-token pooled, L2-normalised, instruction-aware queries) using the pinned `Qwen/Qwen3-Embedding-0.6B` weights",
    "intro": (
        "At inference each text is tokenised with left padding and truncated at 8,192 tokens, a 0.6 B-parameter Qwen3 "
        "decoder encodes it, the hidden state of the **last token** is taken as the text's vector, and the pipeline "
        "L2-normalises it to unit length. Queries are prefixed with a task instruction (`Instruct: …\\nQuery:`) because "
        "the model is instruction-aware; documents are embedded as-is. **Embeddings are representations, not "
        "predictions:** nothing is classified, ranked or decided, and there is no label space. **No adaptation "
        "occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in this "
        "notebook — the upstream checkpoint supplies the weights and tokenizer, and the carried pipeline module adds "
        "snapshot verification, input validation with named ceilings, the query/document formatting contract, a fixed "
        "output contract and the `cosine_similarity`, `validate_inputs` and `evaluation_report` helpers. The default "
        "sample is the four sentences from the pinned upstream README; the cosine values shown for them are a "
        "qualitative check, not a benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, prepare a small identified set of queries and documents and validate it "
        "into an input manifest, embed queries and documents through the public API with the instruction contract, "
        "read the vectors correctly (shape, pooling, normalisation, per-text unit), compare a query with two documents "
        "by cosine as a qualitative check, produce an evaluation report that is honestly `not-measurable` because an "
        "embedding has no intrinsic metric, exercise an optional BYOD path, and export identifiers alongside vectors "
        "plus provenance."
    ),
    "exclusions": (
        "reranking (see the sibling Qwen3 reranker pipeline), text generation or chat, classification, clustering "
        "quality, retrieval evaluation (nDCG/recall need a labelled query–document set), Matryoshka dimension "
        "truncation (fixed at 1024 here), or any training."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available. **Precision differs by device:** the pipeline runs float32 on CPU and bfloat16 on CUDA, so cosine values can differ in the second or third decimal between the two. CPU is slow for large corpora but fine for a handful of sentences: the repository's model card records 6.6 s to load and 0.44 s to embed the four default texts on CPU in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the ~1.19 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and NumPy; what a dense vector, a unit norm and cosine similarity are.",
        "- **Data:** the default sample is four short English sentences (two queries, two documents) taken verbatim from the pinned upstream README, written into the notebook as string literals, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 text file with one document per non-empty line (at most 64 lines, each under 100,000 characters; text beyond 8,192 tokens is truncated and flagged) plus a query typed into the form. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Prepare the sample texts or optional BYOD\n\n"
                "The default sample is **public and bundled in code**: the two queries and two documents from the pinned "
                "upstream README (Apache-2.0), each given a stable identifier (`q1`, `q2`, `d1`, `d2`) so every vector and "
                "similarity can be mapped back to its text. They exist to show the contract, not to measure anything; the "
                "repository's smoke run embedded exactly these and reproduced the README's printed cosine matrix to four "
                "decimals. BYOD is optional and disabled by default; when enabled, upload one UTF-8 text file (one document "
                "per line) and set `BYOD_QUERY`; documents are identified `d1…dN` in file order. The query instruction "
                "(`DEFAULT_QUERY_INSTRUCTION`) is printed because it is part of the query vector: a different instruction "
                "produces a different embedding. Look for a dictionary naming the sample kind, the identifiers, character "
                "counts and the instruction."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_QUERY = 'What is the capital of China?'  # @param {{type:\"string\"}}\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    corpus_name = next(iter(uploaded))\n"
                "    lines = [line.strip() for line in io.TextIOWrapper(io.BytesIO(uploaded[corpus_name]), encoding='utf-8')]\n"
                "    documents = [line for line in lines if line]\n"
                "    queries = [BYOD_QUERY.strip()]\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Public sample: the pinned upstream README's example queries and documents, as string literals.\n"
                "    queries = ['What is the capital of China?', 'Explain gravity']\n"
                "    documents = [\n"
                "        'The capital of China is Beijing.',\n"
                "        'Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.',\n"
                "    ]\n"
                "    corpus_name = 'upstream_readme_example'\n"
                "    sample_kind = 'public (pinned upstream README example, bundled as literals)'\n\n"
                "query_ids = [f'q{{i + 1}}' for i in range(len(queries))]\n"
                "document_ids = [f'd{{i + 1}}' for i in range(len(documents))]\n"
                "corpus_sha256 = hashlib.sha256('\\n'.join(queries + documents).encode('utf-8')).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': corpus_name, 'query_ids': query_ids, 'document_ids': document_ids, 'chars': {{i: len(t) for i, t in zip(query_ids + document_ids, queries + documents, strict=True)}}, 'corpus_sha256': corpus_sha256, 'query_instruction': DEFAULT_QUERY_INSTRUCTION}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the input → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `embed` applies "
                "— both route through the same private `_check_inputs` — so type, batch size 1..`MAX_BATCH`, non-empty "
                "text, character ceiling `MAX_TEXT_CHARS`, a `kind` in `KINDS` and a non-empty instruction are enforced "
                "identically. It returns an **input manifest** naming the schema and ceilings, each input's identifier, "
                "character count and kind, and the verdict. Both halves of the corpus are validated: the documents produce "
                "the manifest, and the queries — which carry the instruction prefix, so their vectors differ from the same "
                "text embedded as a document — are validated the same way and recorded under `query_manifest`. The manifest "
                "is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also "
                "validates a deliberately oversized batch and records the pipeline's own error message as a finding. "
                "Token-level truncation cannot be observed at this stage because it happens inside the tokenizer; the "
                "result's `truncated` flags are read in Section 6."
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_BATCH': MAX_BATCH, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'EMBEDDING_DIM': EMBEDDING_DIM}}}})\n"
                "input_manifest = validate_inputs(documents, 'document', names=document_ids)\n"
                "input_manifest['query_manifest'] = validate_inputs(queries, 'query', names=query_ids)\n"
                "# Demonstrate rejection on an input that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(['probe'] * (MAX_BATCH + 1))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'oversized-batch-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Embed and read the vectors correctly\n\n"
                "`embed(texts, kind, instruction)` returns a dict with `embeddings` — one list per input text, in input "
                "order, each of length `dim` (1024) — plus `pooling` (`last_token`), `normalized` (`True`: every vector has "
                "unit L2 norm), `kind`, the `instruction` applied (queries only, `None` for documents), `n_tokens` per text, "
                "`truncated` flags (a text that hit the 8,192-token ceiling was cut and its vector represents only the kept "
                "prefix), and the model identity. The unit of embedding is **one vector per text**; there is no per-token or "
                "per-chunk output, and a text longer than the window is not chunked for you. Missing data has no meaning "
                "here: empty strings are rejected, not embedded. Below, each query is compared with both documents by cosine "
                "as a **qualitative check** that the contract works: the matching document should score higher than the "
                "unrelated one. Cosine values are similarities in `[-1, 1]` on this model's geometry, not probabilities and "
                "not calibrated; absolute values are not comparable across models, and a threshold for \"relevant\" is the "
                "caller's to set on labelled data. Inference is deterministic on a fixed device and dtype (no sampling, "
                "`torch.inference_mode`); float32 (CPU) and bfloat16 (CUDA) differ in the second or third decimal. As "
                "recorded in the model card, the repository's CPU smoke on these four texts produced the cosine matrix "
                "`[[0.7646, 0.1414], [0.1355, 0.6000]]`, equal to the upstream README's printed values to four decimals; "
                "that is one observation on four sentences, not a retrieval score."
            ),
            "code": (
                "query_result = pipe.embed(queries, kind='query', instruction=DEFAULT_QUERY_INSTRUCTION)\n"
                "document_result = pipe.embed(documents, kind='document')\n"
                "query_vectors = np.asarray(query_result['embeddings'], dtype=np.float32)\n"
                "document_vectors = np.asarray(document_result['embeddings'], dtype=np.float32)\n"
                "print({{'query_shape': query_vectors.shape, 'document_shape': document_vectors.shape, 'dim': document_result['dim'], 'pooling': document_result['pooling'], 'normalized': document_result['normalized'], 'norms': [round(float(v), 4) for v in np.linalg.norm(np.vstack([query_vectors, document_vectors]), axis=1)], 'device': pipe.device}})\n"
                "print({{'n_tokens': dict(zip(query_ids + document_ids, query_result['n_tokens'] + document_result['n_tokens'], strict=True)), 'truncated': dict(zip(query_ids + document_ids, query_result['truncated'] + document_result['truncated'], strict=True))}})\n"
                "if any(query_result['truncated'] + document_result['truncated']):\n"
                "    print('NOTE: at least one text hit MAX_TEXT_TOKENS and was truncated; its vector represents the kept prefix only.')\n"
                "similarity = cosine_similarity(query_result['embeddings'], document_result['embeddings'])\n"
                "for query_id, row in zip(query_ids, similarity, strict=True):\n"
                "    print(query_id, {{document_id: round(value, 4) for document_id, value in zip(document_ids, row, strict=True)}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — even, as "
                "here, when nothing is measurable. **No intrinsic metric exists** for an embedding: the vectors are "
                "representations, the repository ships `cosine_similarity` as a comparison helper and no metric helper, and "
                "so the verdict is always `not-measurable` and the report states what would make the task measurable — for "
                "retrieval, a query–document set with relevance judgements scored by nDCG@k or recall@k; for classification "
                "or clustering, labelled texts and a fitted classifier or cluster assignment. Supplying labels does not "
                "change the verdict, because there is no metric to score them with; the helper records that fact in "
                "`reason` instead of inventing a number. The report covers the document embeddings (the queries are "
                "representations of the same kind, so which half is scored cannot change a `not-measurable` verdict) and is "
                "written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(document_result, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is computed: embeddings are representations; the cosine table above is a qualitative check, and a retrieval or classification score needs labelled data.')"
            ),
        },
        {
            "md": (
                "## 8. Export identifiers alongside vectors, and provenance\n\n"
                "The vectors are written as CSV (`outputs/{stem}_vectors.csv`) with one row per text — `id`, `kind`, "
                "`n_tokens`, `truncated`, then `e0000…e1023` — so every vector stays attached to its identifier for "
                "downstream use. Machine-readable JSON preserves the identified texts, the query instruction, the cosine "
                "table keyed by identifier, the per-text token counts and truncation flags, the corpus digest, the input "
                "manifest, the evaluation report, the notebook's source (repository, revision, embedded module digest, "
                "generator), the model identifier, the immutable model revision, the model licence, and the runtime "
                "identity (Python, `torch`, `transformers`, device and the precision implied by it). No credentials are "
                "recorded."
            ),
            "code": (
                "import csv\n\n"
                "with open('outputs/{stem}_vectors.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['id', 'kind', 'n_tokens', 'truncated'] + [f'e{{i:04d}}' for i in range(EMBEDDING_DIM)])\n"
                "    for kind, ids, result in (('query', query_ids, query_result), ('document', document_ids, document_result)):\n"
                "        for text_id, vector, n_tokens, truncated in zip(ids, result['embeddings'], result['n_tokens'], result['truncated'], strict=True):\n"
                "            writer.writerow([text_id, kind, n_tokens, truncated] + [f'{{value:.7f}}' for value in vector])\n"
                "payload = {{\n"
                "    'texts': {{**dict(zip(query_ids, queries, strict=True)), **dict(zip(document_ids, documents, strict=True))}},\n"
                "    'query_instruction': query_result['instruction'],\n"
                "    'embedding_contract': {{'dim': document_result['dim'], 'pooling': document_result['pooling'], 'normalized': document_result['normalized'], 'unit': 'one vector per text'}},\n"
                "    'n_tokens': dict(zip(query_ids + document_ids, query_result['n_tokens'] + document_result['n_tokens'], strict=True)),\n"
                "    'truncated': dict(zip(query_ids + document_ids, query_result['truncated'] + document_result['truncated'], strict=True)),\n"
                "    'cosine_similarity': {{query_id: dict(zip(document_ids, row, strict=True)) for query_id, row in zip(query_ids, similarity, strict=True)}},\n"
                "    'vectors_file': 'outputs/{stem}_vectors.csv',\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sample': {{'kind': sample_kind, 'name': corpus_name, 'corpus_sha256': corpus_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'precision': 'bfloat16' if pipe.device.startswith('cuda') else 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The vectors are representations of the texts in this model's 1024-dimensional space: they predict nothing, carry "
        "no labels, and their only meaning is relative — cosine between two vectors from the same model and the same "
        "instruction. The cosine table on the default sample shows that the contract works on four short English "
        "sentences; it is not a retrieval score, and it must not be generalised to other languages, domains, long "
        "documents (truncated at 8,192 tokens), or a different query instruction, which changes the query vectors. "
        "Values are uncalibrated similarities, absolute levels are model-specific, and any relevance threshold belongs to "
        "the caller and to labelled data. Vectors from the CPU (float32) and CUDA (bfloat16) paths are close but not "
        "bitwise equal. The evaluation report is `not-measurable` by construction here, which is the honest verdict for "
        "an embedding, not a gap in the notebook. The pipeline provides no reranking, generation, classification, "
        "chunking, dimension truncation, or training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input, execute the public pipeline "
        "path, and emit the shown machine-readable outputs in the tested runtime — without the repository being "
        "reachable. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments:** change the instruction passed to `embed(..., kind='query', instruction=…)` to a "
        "task-specific one (for example a code-search task) and watch the cosine table move; enable `USE_BYOD` with a "
        "small corpus file and your own query, then hand-label which lines are relevant to compute recall@k yourself — "
        "the first step towards a real retrieval number and the labelled data the evaluation report asks for; embed the "
        "same text as `kind='query'` and as `kind='document'` to see how much the instruction prefix shifts a vector.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/qwen3-embedding-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/QwenLM/Qwen3-Embedding\n"
        "- Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models (2025): https://arxiv.org/abs/2506.05176"
    ),
}
