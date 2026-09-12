from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "qwen3-embedding-0.6b"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Contract from the pinned upstream README ("Transformers Usage"): left padding, last-token pooling,
# L2 normalisation, max_length 8192, and an "Instruct: ...\nQuery:" prefix on queries only.
EMBEDDING_DIM = 1024  # hidden_size in the pinned config.json; 1_Pooling/config.json word_embedding_dimension
MAX_TEXT_TOKENS = 8192  # tokenizer truncation length; the model's context is 32768 but the README uses 8192
MAX_TEXT_CHARS = 100_000  # pre-tokenisation guard so a runaway string is rejected before it is tokenised
MAX_BATCH = 64  # texts per embed() call
DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
POOLING = "last_token"
KINDS = ("query", "document")


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest.hexdigest()} != manifest {entry['sha256']}")
    return manifest


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def format_query(query: str, instruction: str = DEFAULT_QUERY_INSTRUCTION) -> str:
    """Upstream `get_detailed_instruct`: queries carry a task instruction, documents do not."""
    return f"Instruct: {instruction}\nQuery:{query}"


def cosine_similarity(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    """Cosine similarity matrix between two lists of vectors (no metric: there is no ground truth)."""
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("inputs must be 2-D with the same embedding dimension")
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    return (x @ y.T).tolist()


@dataclass
class Qwen3EmbeddingPipeline:
    """Text embedder. `_runner` maps formatted texts to (pooled un-normalised vectors, token counts)."""

    _runner: Callable[[list[str]], tuple[np.ndarray, list[int]]]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Qwen3EmbeddingPipeline:
        import torch
        from transformers import AutoModel, AutoTokenizer

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), dict(local_files_only=True)
        elif allow_download:
            source, kwargs = MODEL_ID, dict(revision=MODEL_REVISION)
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        tokenizer = AutoTokenizer.from_pretrained(
            source, padding_side="left", trust_remote_code=False, **kwargs
        )
        model = AutoModel.from_pretrained(source, dtype=dtype, trust_remote_code=False, **kwargs)
        model = model.to(resolved_device).eval()

        def runner(texts: list[str]) -> tuple[np.ndarray, list[int]]:
            batch = tokenizer(
                texts, padding=True, truncation=True, max_length=MAX_TEXT_TOKENS, return_tensors="pt"
            )
            batch = batch.to(resolved_device)
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
            pooled = hidden[:, -1]  # left padding: the last position is the last real token of every row
            counts = batch["attention_mask"].sum(dim=1).tolist()
            return pooled.float().cpu().numpy(), [int(c) for c in counts]

        return cls(runner, resolved_device)

    def embed(
        self,
        texts: Sequence[str],
        kind: str = "document",
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> dict[str, Any]:
        """Embed up to MAX_BATCH texts. `kind="query"` prepends the instruction; documents get none."""
        if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a list of str, not a single string")
        if not 1 <= len(texts) <= MAX_BATCH:
            raise ValueError(f"texts must hold 1..{MAX_BATCH} items, got {len(texts)}")
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{i}] must be str, got {type(text).__name__}")
            if not text.strip():
                raise ValueError(f"texts[{i}] is empty")
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError(f"texts[{i}] has {len(text)} chars; ceiling is {MAX_TEXT_CHARS}")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty str")

        formatted = [format_query(t, instruction) if kind == "query" else t for t in texts]
        pooled, n_tokens = self._runner(formatted)
        pooled = np.asarray(pooled, dtype=np.float32)
        if pooled.shape != (len(texts), EMBEDDING_DIM):
            raise RuntimeError(f"backend returned {pooled.shape}, expected ({len(texts)}, {EMBEDDING_DIM})")
        normalized = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return {
            "embeddings": normalized.tolist(),
            "dim": EMBEDDING_DIM,
            "pooling": POOLING,
            "normalized": True,
            "kind": kind,
            "instruction": instruction if kind == "query" else None,
            "n_tokens": list(n_tokens),
            "truncated": [n >= MAX_TEXT_TOKENS for n in n_tokens],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
