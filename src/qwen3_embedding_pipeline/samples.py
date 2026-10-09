"""Query–positive pair dataset contract for contrastive adaptation of the embedder: the pinned Banking77
sample, validation, seeded splitting, BYOD loaders and CSV export.

The default dataset is **real** and a retrieval task the embedder was not tuned for: Banking77 (Casanueva et
al., 2020; CC BY 4.0), 13,083 customer-support messages labelled with 77 fine-grained banking intents. Two CSV
files (`train.csv`, `test.csv`) are fetched from the PolyAI `task-specific-datasets` repository at a pinned
commit and refused on any byte-size or SHA-256 mismatch. Every intent name becomes a short **document**
(`card_arrival` → `card arrival`); each message is a **query** whose positive document is its intent phrase,
so retrieval over the 77 documents is intent detection by nearest neighbour. Training and validation queries
are drawn from `train.csv`, test queries from `test.csv` — the release's own partition — balanced over the
77 intents.

A record is ``{id, query, positive}`` (an optional ``negative`` is accepted and carried); the document set
of a dataset is its sorted unique positives.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import re
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .pipeline import MAX_TEXT_CHARS, MODEL_ID

CORPUS_NAME = "Banking77 (messages → intent phrases)"
CORPUS_RELEASE = "PolyAI-LDN/task-specific-datasets @ 57ec275d8078af65b7731c2a98be812d844a6d6b"
CORPUS_BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data/"
)
CORPUS_FILES = {
    "train": ("train.csv", 839_073, "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "test": ("test.csv", 239_961, "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
}
CORPUS_LICENSE = "CC BY 4.0 (Casanueva et al. 2020; PolyAI-LDN/task-specific-datasets)"
CORPUS_ROWS = {"train": 10_003, "test": 3_080}
CORPUS_INTENTS = 77
DEFAULT_CACHE_DIR = Path("weights") / "banking77"
DEFAULT_INSTRUCTION = "Given a customer support message, retrieve the banking intent it expresses"
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 616, "validation": 154, "test": 385}  # 8 / 2 / 5 per intent, balanced over 77
MIN_RECORDS = 8
MAX_RECORDS = 20_000
MAX_DOCUMENT_CHARS = 1_000
MIN_DOCUMENTS = 2
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def intent_phrase(intent: str) -> str:
    """The document text of an intent: its snake_case name as words (`card_arrival` → `card arrival`)."""
    return " ".join(intent.strip().split("_"))


def fetch_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> dict[str, bytes]:
    """Return the two pinned Banking77 CSVs (bytes) from the cache or the project repository, verified."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, (name, size, digest) in CORPUS_FILES.items():
        local = cache / name
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = CORPUS_BASE_URL + name
            if fetcher is not None:
                data = fetcher(url)
            else:
                with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (pinned https URL)
                    data = response.read()
            if len(data) != size or _sha256_bytes(data) != digest:
                raise ValueError(
                    f"{name}: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}…, "
                    f"pinned {size} / {digest[:16]}…"
                )
            local.write_bytes(data)
        out[split] = data
    return out


def read_corpus(files: Mapping[str, bytes]) -> dict[str, list[dict[str, Any]]]:
    """Parse the CSV members (columns `text`, `category`) into flat records keeping the raw intent name."""
    out = {}
    for split in CORPUS_FILES:
        if split not in files:
            raise ValueError(f"corpus is missing the {split} file")
        rows = list(csv.DictReader(io.StringIO(files[split].decode("utf-8"))))
        if not rows or {"text", "category"} - set(rows[0]):
            raise ValueError(f"{split}: expected columns text and category")
        if len(rows) != CORPUS_ROWS[split]:
            raise ValueError(f"{split}: {len(rows)} rows, expected {CORPUS_ROWS[split]}")
        out[split] = [
            {"id": f"{split}-{i:05d}", "text": r["text"].strip(), "intent": r["category"].strip()}
            for i, r in enumerate(rows)
        ]
        intents = {r["intent"] for r in out[split]}
        if len(intents) != CORPUS_INTENTS:
            raise ValueError(f"{split}: {len(intents)} intents, expected {CORPUS_INTENTS}")
    return out


def filter_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn corpus rows into query–positive pairs; drop empty, over-long and repeated queries."""
    seen: set[str] = set()
    kept = []
    for record in records:
        query = str(record["text"]).strip()
        key = query.lower()
        if not query or key in seen or len(query) > MAX_TEXT_CHARS:
            continue
        seen.add(key)
        intent = str(record["intent"])
        kept.append({"id": record["id"], "query": query, "positive": intent_phrase(intent), "intent": intent})
    return kept


def build_sample_dataset(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Balanced seeded draws over all 77 intents: training and validation from `train` (disjoint queries),
    test from `test`."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    for name, size in sizes.items():
        if size % CORPUS_INTENTS:
            raise ValueError(f"{name} size {size} is not a multiple of the {CORPUS_INTENTS} intents")
    rng = random.Random(seed)
    pools = {"train": filter_records(corpus["train"]), "test": filter_records(corpus["test"])}
    intents = sorted({r["intent"] for r in pools["train"]})
    by_intent = {
        split: {intent: [r for r in pool if r["intent"] == intent] for intent in intents}
        for split, pool in pools.items()
    }
    for split in by_intent.values():
        for records in split.values():
            rng.shuffle(records)
    cursor = dict.fromkeys(intents, 0)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, size in sizes.items():
        source = "test" if name == "test" else "train"
        per_intent = size // CORPUS_INTENTS
        picked = []
        for intent in intents:
            pool = by_intent[source][intent]
            start = cursor[intent] if source == "train" else 0
            chunk = pool[start : start + per_intent]
            if len(chunk) < per_intent:
                raise ValueError(
                    f"{name}: only {len(chunk)} records available for {intent!r}, need {per_intent}"
                )
            picked.extend(chunk)
            if source == "train":
                cursor[intent] = start + per_intent
        rng.shuffle(picked)
        out[name] = [
            {"id": f"{name}-{i:04d}", "query": r["query"], "positive": r["positive"], "intent": r["intent"]}
            for i, r in enumerate(picked)
        ]
    return out


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    fetcher: Any = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned corpus."""
    return build_sample_dataset(
        read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher)), seed=seed, sizes=sizes
    )


def _check_text(value: Any, label: str, ceiling: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not value.strip():
        raise ValueError(f"{label} is empty")
    if len(value) > ceiling:
        raise ValueError(f"{label} has {len(value)} chars; ceiling is {ceiling}")
    return value.strip()


def _check_record(record: Any, index: int) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/query/positive")
    for key in ("id", "query", "positive"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    item = {
        "id": rid,
        "query": _check_text(record["query"], f"{label}: query", MAX_TEXT_CHARS),
        "positive": _check_text(record["positive"], f"{label}: positive", MAX_DOCUMENT_CHARS),
    }
    if record.get("negative") not in (None, ""):
        item["negative"] = _check_text(record["negative"], f"{label}: negative", MAX_DOCUMENT_CHARS)
        if item["negative"] == item["positive"]:
            raise ValueError(f"{label}: negative equals positive")
    if "intent" in record:
        item["intent"] = str(record["intent"])
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a query–positive dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, query, positive} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    queries: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        queries.add(item["query"].lower())
        checked.append(item)
    docs = documents(checked)
    if len(docs) < MIN_DOCUMENTS:
        raise ValueError(f"a dataset needs at least {MIN_DOCUMENTS} distinct positives; found {len(docs)}")
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_queries": len(queries),
        "n_documents": len(docs),
        "query_chars": {
            "min": min(len(r["query"]) for r in checked),
            "max": max(len(r["query"]) for r in checked),
        },
        "with_negative": sum("negative" in r for r in checked),
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def documents(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The sorted unique positive (and explicit negative) documents of a dataset — the candidates."""
    return sorted(
        {str(r["positive"]).strip() for r in records}
        | {str(r["negative"]).strip() for r in records if r.get("negative")}
    )


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], r["query"], r["positive"], r.get("negative", "")] for r in records]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


_KEY_SPLIT_RE = re.compile(r"[^0-9a-z]+")
NEAR_DUPLICATE_JACCARD = 0.8


def query_key(text: str) -> str:
    """The leakage key of a query: lower-cased, with punctuation and whitespace runs collapsed, so
    "Has my top-up been cancelled?" and "has my top up been cancelled" are the same message."""
    return " ".join(_KEY_SPLIT_RE.split(str(text).lower())).strip()


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no query appears in two splits (leakage check): exact repeats after lower-casing and
    collapsing punctuation and whitespace (`query_key`). Near-paraphrases are reported by
    `near_duplicate_pairs`, not refused."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = query_key(record["query"])
            if key in seen and seen[key] != name:
                raise ValueError(f"query {record['query'][:60]!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating queries."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records, min_records=1)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = query_key(record["query"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    if not _split_sizes_ok(len(unique), val_fraction, test_fraction):
        needed = min_split_records(val_fraction=val_fraction, test_fraction=test_fraction)
        raise ValueError(
            f"the dataset has {len(unique)} distinct queries (of {len(checked)} records); the split keeps "
            f"{test_fraction:.0%} for test and {val_fraction:.0%} for validation (each scored over at least "
            f"{MIN_DOCUMENTS} records) and needs {MIN_RECORDS} for training, so at least {needed} distinct "
            "queries are required"
        )
    random.Random(seed).shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    n_val = round(len(unique) * val_fraction)
    # Each evaluated split needs two distinct documents to rank (MIN_DOCUMENTS). When the seeded order
    # puts a single document in one, the nearest later record with a different document is moved in.
    _spread_documents(unique, 0, n_test)
    _spread_documents(unique, n_test, n_val)
    splits = {
        "test": unique[:n_test],
        "validation": unique[n_test : n_test + n_val],
        "train": unique[n_test + n_val :],
    }
    for name, part in splits.items():
        if part and len(documents(part)) < MIN_DOCUMENTS:
            raise ValueError(
                f"the {name} split ({len(part)} records) holds 1 distinct document; add records so that "
                f"at least {MIN_DOCUMENTS} distinct positives each have several queries"
            )
    return splits


def _split_sizes_ok(n: int, val_fraction: float, test_fraction: float) -> bool:
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    return (
        n - n_test - n_val >= MIN_RECORDS
        and n_test >= MIN_DOCUMENTS
        and (n_val == 0 or n_val >= MIN_DOCUMENTS)
    )


def min_split_records(*, val_fraction: float = 0.15, test_fraction: float = 0.2) -> int:
    """The smallest number of distinct queries `split_dataset` accepts: MIN_RECORDS left for training and
    at least MIN_DOCUMENTS records in each evaluated split (12 at the default fractions)."""
    for n in range(MIN_RECORDS, MAX_RECORDS + 1):
        if _split_sizes_ok(n, val_fraction, test_fraction):
            return n
    raise ValueError("no dataset size satisfies these fractions")


def _spread_documents(records: list[dict[str, Any]], start: int, size: int) -> None:
    """Give records[start:start+size] MIN_DOCUMENTS distinct documents when a later record allows it."""
    if size < MIN_DOCUMENTS:
        return
    present = set(documents(records[start : start + size]))
    if len(present) >= MIN_DOCUMENTS:
        return
    for j in range(start + size, len(records)):
        if set(documents([records[j]])) - present:
            last = start + size - 1
            records[last], records[j] = records[j], records[last]
            return


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _KEY_SPLIT_RE.split(str(text).lower()) if t)


def near_duplicate_pairs(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    threshold: float = NEAR_DUPLICATE_JACCARD,
    evaluated: Sequence[str] = ("validation", "test"),
    max_examples: int = 5,
) -> dict[str, Any]:
    """Report (not refuse) each query of an evaluated split that is a near-paraphrase of a query in another
    split: token Jaccard >= threshold. Exact repeats are already refused by `check_split_disjoint`."""
    pool = [(name, _tokens(r["query"]), str(r["query"])) for name, records in splits.items() for r in records]
    frequency: dict[str, int] = {}
    for _name, toks, _query in pool:
        for tok in toks:
            frequency[tok] = frequency.get(tok, 0) + 1

    # Prefix filtering: two token sets with Jaccard >= t share at least one of the
    # len(a) - ceil(t * len(a)) + 1 rarest tokens of each, so only those are indexed and probed.
    def prefix(toks: frozenset[str]) -> list[str]:
        ordered = sorted(toks, key=lambda tok: (frequency[tok], tok))
        return ordered[: len(ordered) - math.ceil(threshold * len(ordered)) + 1]

    index: dict[str, list[int]] = {}
    for i, (_name, toks, _query) in enumerate(pool):
        for tok in prefix(toks):
            index.setdefault(tok, []).append(i)
    counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for name in evaluated:
        found = 0
        for own, toks, query in pool:
            if own != name or not toks:
                continue
            best = None
            for i in sorted({i for tok in prefix(toks) for i in index.get(tok, [])}):
                other_name, other_toks, other_query = pool[i]
                if other_name == name:
                    continue
                score = len(toks & other_toks) / len(toks | other_toks)
                if score >= threshold and (best is None or score > best[0]):
                    best = (score, other_name, other_query)
            if best is not None:
                found += 1
                if len(examples) < max_examples:
                    examples.append(
                        {"split": name, "query": query, "other_split": best[1], "other_query": best[2],
                         "jaccard": round(best[0], 3)}
                    )
        counts[name] = found
    return {"threshold": threshold, "near_duplicates": counts, "examples": examples}


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read `{id, query, positive[, negative]}` records from CSV (columns id, query, positive and an optional
    negative), a JSON array or JSONL."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        missing = {"id", "query", "positive"} - set(rows[0].keys() if rows else set())
        if missing:
            raise ValueError(f"CSV is missing columns {sorted(missing)}")
        out = []
        for r in rows:
            item = {"id": r["id"], "query": r["query"], "positive": r["positive"]}
            if r.get("negative"):
                item["negative"] = r["negative"]
            out.append(item)
        return out
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .csv, .json or .jsonl")


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "query", "positive", "negative"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "id": record["id"],
                    "query": record["query"],
                    "positive": record["positive"],
                    "negative": record.get("negative", ""),
                }
            )
    return out
