"""Query–positive pair dataset contract for contrastive adaptation of the embedder: the pinned Banking77
sample, validation, seeded splitting, BYOD loaders and CSV export.

The default dataset is **real**: Banking77 (Casanueva et al., 2020; CC BY 4.0), 13,083 customer-support
messages labelled with 77 fine-grained banking intents. Two CSV files (`train.csv`, `test.csv`) are fetched
from the PolyAI `task-specific-datasets` repository at a pinned commit and refused on any byte-size or
SHA-256 mismatch. Every intent name becomes a short **document** (`card_arrival` → `card arrival`); each
message is a **query** whose positive document is its intent phrase, so retrieval over the 77 documents is
intent detection by nearest neighbour. Training and validation queries are drawn from `train.csv`, test
queries from `test.csv` — the release's own partition — balanced over the 77 intents.

DAT9 (training-data overlap): Banking77 is an MTEB classification task, and Qwen3-Embedding reports MTEB
results (the upstream evaluation/task_prompts.json carries a Banking77Classification prompt). The technical
report (arXiv:2506.05176, Table 6) neither lists nor excludes Banking77 from the training mix, so the frozen
model's score here may be in-distribution.

The 385 seed-42 test queries (5 per intent) were compared across configurations in 2026-09, so they are the
**exposed development test**. The **fresh test** (`build_fresh_test`) is 10 per intent drawn from the
test.csv messages this repository had never scored: not the tutorial's 385, not the workshop W16's 154, and
not an exact repeat of any train.csv message. It is pinned by a manifest SHA-256 and scored only under the
pre-registered settings (docs/qwen3-embedding-preregistration.md).

A record is ``{id, query, positive}`` (an optional ``negative`` is accepted and carried); the document set of
a dataset is its sorted unique positives."""

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


# ---- Fresh test, pre-registration and the validation grid (row 72 fix, 2026-10-17) ------------------------

WORKSHOP_W16_SEED = (
    42  # tools/semantic_search_reranking_workshop_source.py: SAMPLE_SEED, 2 per intent from test.csv
)
WORKSHOP_W16_PER_INTENT = 2
FRESH_TEST_SEED = 20_261_017
FRESH_TEST_PER_INTENT = 10
FRESH_TEST_SIZE = FRESH_TEST_PER_INTENT * CORPUS_INTENTS  # 770
FRESH_TEST_MANIFEST_SHA256 = "fe68b43ae56b9092ab92f6eeb610b2179222d1dfa04c3eebd9c658ac69c0908f"


class DatasetRefused(ValueError):
    """A BYOD input the notebook refuses (DAT20): shown as one message, not a traceback."""

    dimer_refusal = True

    def _render_traceback_(self) -> list[str]:  # IPython shows the message only
        return [str(self)]


def workshop_w16_queries(corpus: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """The 154 test.csv messages the merged workshop W16 scores (2 per intent, its own seeded draw),
    reproduced with the workshop's logic so the fresh test can exclude them."""
    test_rows = [
        r
        for r in corpus["test"]
        if str(r["text"]).strip()
        and str(r["intent"]).strip()
        and len(str(r["text"]).strip()) <= MAX_TEXT_CHARS
    ]
    intents = sorted({str(r["intent"]).strip() for r in corpus["train"] if str(r["text"]).strip()})
    by_intent: dict[str, list[Mapping[str, Any]]] = {}
    for row in test_rows:
        by_intent.setdefault(str(row["intent"]).strip(), []).append(row)
    rng = random.Random(WORKSHOP_W16_SEED)
    out = []
    for intent in intents:
        pool = list(by_intent.get(intent, []))
        rng.shuffle(pool)
        out.extend(
            {"id": r["id"], "query": str(r["text"]).strip(), "intent": intent}
            for r in pool[:WORKSHOP_W16_PER_INTENT]
        )
    return out


FRESH_TEST_2_SEED = 20_261_018
FRESH_TEST_2_MANIFEST_SHA256 = "cd0171285dfd6ccc72b88d82b43ad856af6869a84a49cc50ac06046119f37a63"
FRESH_TEST_1_LABEL = "fresh test 1 (scored once, 2026-10-17, bf16 recipe; now an exposed test)"


def fresh_test_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Manifest SHA-256 of a fresh-test draw: the sorted (source id, query) pairs."""
    payload = sorted([str(r["source_id"]), str(r["query"])] for r in records)
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def build_fresh_test(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """The pinned fresh test: 10 per intent from the test.csv messages this repository never scored.

    Excluded before the draw: the tutorial's seed-42 test draw (the exposed development test), the workshop
    W16's 154 queries and every exact repeat (`query_key`) of any train.csv message. Near-paraphrases (token
    Jaccard >= 0.8) of the training and validation draw are reported, not removed. `verify` refuses a draw
    whose manifest SHA-256 is not FRESH_TEST_MANIFEST_SHA256."""
    exposed_splits = build_sample_dataset(corpus, seed=SAMPLE_SEED)
    tutorial_keys = {query_key(r["query"]) for r in exposed_splits["test"]}
    workshop_keys = {query_key(r["query"]) for r in workshop_w16_queries(corpus)}
    exposed = tutorial_keys | workshop_keys
    train_keys = {query_key(r["query"]) for r in filter_records(corpus["train"])}
    pool = filter_records(corpus["test"])
    never_scored = [r for r in pool if query_key(r["query"]) not in exposed]
    exact_repeats = [r for r in never_scored if query_key(r["query"]) in train_keys]
    clean = [r for r in never_scored if query_key(r["query"]) not in train_keys]
    intents = sorted({r["intent"] for r in pool})
    rng = random.Random(FRESH_TEST_SEED)
    picked = []
    for intent in intents:
        candidates = sorted((r for r in clean if r["intent"] == intent), key=lambda r: r["id"])
        rng.shuffle(candidates)
        if len(candidates) < FRESH_TEST_PER_INTENT:
            raise ValueError(f"fresh test: only {len(candidates)} never-scored messages for {intent!r}")
        picked.extend(candidates[:FRESH_TEST_PER_INTENT])
    rng.shuffle(picked)
    records = [
        {
            "id": f"fresh-{i:04d}",
            "source_id": r["id"],
            "query": r["query"],
            "positive": r["positive"],
            "intent": r["intent"],
        }
        for i, r in enumerate(picked)
    ]
    keys = {query_key(r["query"]) for r in records}
    if keys & exposed or keys & train_keys or len(keys) != len(records):
        raise ValueError("fresh test overlaps an exposed or training message")
    digest = fresh_test_digest(records)
    if verify and digest != FRESH_TEST_MANIFEST_SHA256:
        raise ValueError(
            f"fresh test manifest {digest[:16]}... differs from the pinned "
            f"{FRESH_TEST_MANIFEST_SHA256[:16]}..."
        )
    reference = splits or exposed_splits
    near = near_duplicate_pairs(
        {"train": reference["train"], "validation": reference["validation"], "fresh_test": records},
        evaluated=("fresh_test",),
    )
    pool_near = near_duplicate_pairs(
        {"train": reference["train"], "validation": reference["validation"], "pool": clean},
        evaluated=("pool",),
        max_examples=0,
    )
    return {
        "records": records,
        "manifest": {
            "source": f"test.csv at {CORPUS_RELEASE} (sha256 {CORPUS_FILES['test'][2]})",
            "seed": FRESH_TEST_SEED,
            "per_intent": FRESH_TEST_PER_INTENT,
            "n": len(records),
            "test_pool_after_filter": len(pool),
            "exposed_tutorial": len(tutorial_keys),
            "exposed_workshop_w16": len(workshop_keys),
            "exposed_union": len(exposed),
            "never_scored": len(never_scored),
            "exact_repeats_of_train_csv_dropped": len(exact_repeats),
            "near_duplicates_vs_train_validation": {
                "threshold_jaccard": NEAR_DUPLICATE_JACCARD,
                "in_fresh_test": near["near_duplicates"]["fresh_test"],
                "in_clean_pool": pool_near["near_duplicates"]["pool"],
                "examples": near["examples"],
                "reported_not_removed": True,
            },
            "manifest_sha256": digest,
        },
    }


# The pre-registered settings (docs/qwen3-embedding-preregistration.md). A run scores the fresh test only when
# every one of them holds and the configuration is the one the validation grid selected in the same session.
PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data_source": "banking77@57ec275d8078af65b7731c2a98be812d844a6d6b",
    "split_seed": SAMPLE_SEED,
    "sizes": {
        "train": 616,
        "validation": 154,
        "exposed_development_test": 385,
        "fresh_test": FRESH_TEST_SIZE,
    },
    "fresh_test_sha256": FRESH_TEST_2_MANIFEST_SHA256,
    "precision": "float32",
    "instruction": DEFAULT_INSTRUCTION,
    "epochs": 2,
    "batch_size": 16,
    "temperature": 0.05,
    "max_train_tokens": 64,
    "adapt_seed": 0,
    "optimizer": "AdamW(weight_decay=0.01), clip 1.0, no scheduler",
    "epoch_selection": "highest validation MRR",
    "grid": "C0/C1/C2 on validation MRR, margin 0.02",
}
GRID_CONFIGS: dict[str, dict[str, Any]] = {
    "C0": {"trainable_layers": 2, "lr": 5e-5},
    "C1": {"trainable_layers": 2, "lr": 2e-5},
    "C2": {"trainable_layers": 4, "lr": 2e-5},
}
GRID_MARGIN = 0.02
# The evaluation of record per recipe (docs/release-verification.md). Recipe 1 (bf16 CUDA training) was
# scored on fresh test 1 by the hosted T4 run of 2026-10-17; recipe 2 (float32, fresh test 2) is None until
# its first hosted pre-registered run is recorded. Later identical runs of a recipe are reproductions.
RECIPE_1 = "recipe 1 (bf16 CUDA training; fresh test 1)"
RECIPE = "recipe 2 (float32 on every device; fresh test 2)"
RECORDED_EVALUATIONS: dict[str, dict[str, Any] | None] = {
    RECIPE_1: {
        "executor": "Colab CLI 0.7.4, fresh Colab Tesla T4, cuda:0 bfloat16",
        "date": "2026-10-17",
        "commit": "1cc43edb4e4c7ade0df8b3e71e1098d7d50a6119",
        "blob": "971187e6a67b878a6ce11c19c0b297c9d512d719",
        "fresh_test_sha256": FRESH_TEST_MANIFEST_SHA256,
        "fresh_mrr": {"frozen": 0.7338, "adapted": 0.8455},
        "paired_delta_mrr": 0.1118,
        "ci95_queries": [0.0911, 0.134],
        "ci95_intents": [0.0711, 0.1569],
    },
    RECIPE: None,
}
RECORDED_EVALUATION: dict[str, Any] | None = RECORDED_EVALUATIONS[RECIPE]


def select_configuration(
    val_mrr: Mapping[str, float], frozen_val_mrr: float, *, margin: float = GRID_MARGIN
) -> dict[str, Any]:
    """The pre-registered selection rule over best-epoch validation MRR.

    C0 (2 layers @ 5e-5, the current default) is kept unless a challenger beats it by `margin`; C2 (4 layers,
    twice the adapter) must also beat the best 2-layer configuration by `margin`. Within the margin, or on a
    tie, the smaller/current configuration stays. The frozen model is kept unless the selected configuration
    beats the frozen validation MRR by `margin`."""
    missing = set(GRID_CONFIGS) - set(val_mrr)
    if missing:
        raise ValueError(f"grid results are missing {sorted(missing)}")
    selected, reason = "C0", "no challenger beat C0 by the margin"
    if val_mrr["C1"] >= val_mrr["C0"] + margin:
        selected, reason = "C1", f"C1 beat C0 by >= {margin}"
    best_two = max(val_mrr["C0"], val_mrr["C1"])
    if val_mrr["C2"] >= val_mrr["C0"] + margin and val_mrr["C2"] >= best_two + margin:
        selected, reason = "C2", f"C2 beat C0 and the best 2-layer configuration by >= {margin}"
    keep_frozen = val_mrr[selected] < frozen_val_mrr + margin
    return {
        "config": selected,
        "settings": dict(GRID_CONFIGS[selected]),
        "reason": reason,
        "keep_frozen": keep_frozen,
        "selected_model": "frozen" if keep_frozen else f"adapted {selected}",
        "val_mrr": {k: round(float(v), 6) for k, v in val_mrr.items()},
        "frozen_val_mrr": round(float(frozen_val_mrr), 6),
        "margin": margin,
    }


def preregistration_diff(settings: Mapping[str, Any], selection: Mapping[str, Any] | None) -> list[str]:
    """The pre-registered settings a run does not hold (empty = pre-registered). `settings` also carries the
    run's `configuration` ({trainable_layers, lr}); a missing setting counts as a difference, and so does a
    session in which the validation grid did not run."""
    differs = [key for key, value in PREREGISTERED_SETTINGS.items() if settings.get(key, object()) != value]
    if selection is None:
        differs.append("grid (not run in this session)")
    elif settings.get("configuration", object()) != selection["settings"]:
        differs.append("configuration")
    return differs


def test_look(
    preregistered: bool, prior_looks: int, recorded: Mapping[str, Any] | None = RECORDED_EVALUATION
) -> str:
    """SCORE_ONCE item 5 label of a pass. A runtime cannot know about other runtimes: the first pre-registered
    scoring in a runtime is "pre-registered evaluation (this runtime)" unless a hosted evaluation is already
    recorded; later identical scorings are reproductions."""
    if not preregistered:
        return "not scored (exploratory)"
    if prior_looks or recorded is not None:
        return "reproduction"
    return "pre-registered evaluation (this runtime)"


test_look.__test__ = False  # not a pytest test


# ---- Transfer check (optional, display only; R72-M2 ruling) ------------------------------------------------

TRANSFER_NAME = "CLINC150 (Larson et al., 2019): in-scope test queries -> intent phrases"
TRANSFER_RELEASE = "clinc/oos-eval @ 828f8093932c8fe6ca7936c3d2e52903b1c523de"
TRANSFER_URL = "https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json"
TRANSFER_FILE = (
    "data_full.json",
    2_495_390,
    "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
)
TRANSFER_LICENSE = "CC BY 3.0 (Larson et al. 2019; clinc/oos-eval LICENSE)"
TRANSFER_INTENTS = 150
TRANSFER_PER_INTENT = 2
TRANSFER_SEED = 20_261_017
TRANSFER_INSTRUCTION = "Given a request to a virtual assistant, retrieve the intent it expresses"
TRANSFER_MANIFEST_SHA256 = "2f45326ae57c93c79a65f090be7a7623507415298613a15eb6553f960cfb9577"
# The `banking` and `credit_cards` domains of CLINC150 (data/domains.json at the pinned commit, sha256
# b947b579d3b8e74b06f93b01083d8efaff2888b43a3e362533bd88a6e1211b3a): near Banking77's own domain, so the
# transfer check reports them apart from the other 8 domains (peer review F1, 2026-10-18).
TRANSFER_NEAR_DOMAINS = ("banking", "credit_cards")
TRANSFER_NEAR_INTENTS = frozenset(
    (
        "account_blocked",
        "application_status",
        "apr",
        "balance",
        "bill_balance",
        "bill_due",
        "card_declined",
        "credit_limit",
        "credit_limit_change",
        "credit_score",
        "damaged_card",
        "expiration_date",
        "freeze_account",
        "improve_credit_score",
        "interest_rate",
        "international_fees",
        "min_payment",
        "new_card",
        "order_checks",
        "pay_bill",
        "pin_change",
        "redeem_rewards",
        "replacement_card_duration",
        "report_fraud",
        "report_lost_card",
        "rewards_balance",
        "routing",
        "spending_history",
        "transactions",
        "transfer",
    )
)
TRANSFER_GROUPS = ("near-domain (CLINC banking + credit_cards)", "other 8 domains")


def fetch_transfer_corpus(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> bytes:
    """The pinned CLINC150 data_full.json (bytes), refused on any size or SHA-256 mismatch."""
    name, size, digest = TRANSFER_FILE
    cache = Path(cache_dir) if cache_dir is not None else Path("weights") / "clinc150"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / name
    data = local.read_bytes() if local.is_file() else b""
    if len(data) != size or _sha256_bytes(data) != digest:
        if fetcher is not None:
            data = fetcher(TRANSFER_URL)
        else:
            with urllib.request.urlopen(TRANSFER_URL, timeout=120) as response:  # noqa: S310 (pinned https URL)
                data = response.read()
        if len(data) != size or _sha256_bytes(data) != digest:
            raise ValueError(
                f"{name}: fetched {len(data)} bytes with sha256 {_sha256_bytes(data)[:16]}..., "
                f"pinned {size} / {digest[:16]}..."
            )
        local.write_bytes(data)
    return data


def build_transfer_set(data: bytes, *, verify: bool = True) -> dict[str, Any]:
    """2 in-scope test queries per CLINC150 intent (300) and the 150 intent phrases, by a seeded draw."""
    rows = json.loads(data.decode("utf-8"))["test"]
    by_intent: dict[str, list[tuple[int, str]]] = {}
    for i, (text, intent) in enumerate(rows):
        by_intent.setdefault(str(intent), []).append((i, str(text).strip()))
    if len(by_intent) != TRANSFER_INTENTS:
        raise ValueError(f"CLINC150 test: {len(by_intent)} intents, expected {TRANSFER_INTENTS}")
    rng = random.Random(TRANSFER_SEED)
    records = []
    for intent in sorted(by_intent):
        pool = sorted(by_intent[intent])
        rng.shuffle(pool)
        for index, text in pool[:TRANSFER_PER_INTENT]:
            records.append(
                {
                    "id": f"clinc-test-{index:04d}",
                    "query": text,
                    "positive": intent_phrase(intent),
                    "intent": intent,
                }
            )
    docs = documents(records)
    if len(docs) != TRANSFER_INTENTS:
        raise ValueError("CLINC150 intent phrases are not unique")
    digest = dataset_digest(records)
    if verify and digest != TRANSFER_MANIFEST_SHA256:
        raise ValueError(
            f"transfer set manifest {digest[:16]}... differs from the pinned "
            f"{TRANSFER_MANIFEST_SHA256[:16]}..."
        )
    groups = {}
    for label in TRANSFER_GROUPS:
        near = label == TRANSFER_GROUPS[0]
        part = [r for r in records if (r["intent"] in TRANSFER_NEAR_INTENTS) == near]
        groups[label] = {"records": part, "documents": documents(part)}
    return {"records": records, "documents": docs, "manifest_sha256": digest, "groups": groups}


def build_fresh_test_2(
    corpus: Mapping[str, Sequence[Mapping[str, Any]]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Fresh test 2 (pre-registration amendment 1): 10 per intent from the test.csv messages no run has
    scored.

    Excluded: the exposed development test (385), the workshop W16's 154, every exact `query_key` repeat of a
    train.csv message, and fresh test 1 (770, scored once on 2026-10-17). `verify` refuses any draw whose
    manifest SHA-256 is not FRESH_TEST_2_MANIFEST_SHA256."""
    exposed_splits = build_sample_dataset(corpus, seed=SAMPLE_SEED)
    fresh_1 = build_fresh_test(corpus, exposed_splits, verify=verify)["records"]
    excluded = (
        {query_key(r["query"]) for r in exposed_splits["test"]}
        | {query_key(r["query"]) for r in workshop_w16_queries(corpus)}
        | {query_key(r["query"]) for r in filter_records(corpus["train"])}
        | {query_key(r["query"]) for r in fresh_1}
    )
    pool = filter_records(corpus["test"])
    clean = [r for r in pool if query_key(r["query"]) not in excluded]
    rng = random.Random(FRESH_TEST_2_SEED)
    picked = []
    for intent in sorted({r["intent"] for r in pool}):
        candidates = sorted((r for r in clean if r["intent"] == intent), key=lambda r: r["id"])
        rng.shuffle(candidates)
        if len(candidates) < FRESH_TEST_PER_INTENT:
            raise ValueError(f"fresh test 2: only {len(candidates)} never-scored messages for {intent!r}")
        picked.extend(candidates[:FRESH_TEST_PER_INTENT])
    rng.shuffle(picked)
    records = [
        {
            "id": f"fresh2-{i:04d}",
            "source_id": r["id"],
            "query": r["query"],
            "positive": r["positive"],
            "intent": r["intent"],
        }
        for i, r in enumerate(picked)
    ]
    keys = {query_key(r["query"]) for r in records}
    if keys & excluded or len(keys) != len(records):
        raise ValueError("fresh test 2 overlaps an exposed, training or fresh-test-1 message")
    digest = fresh_test_digest(records)
    if verify and digest != FRESH_TEST_2_MANIFEST_SHA256:
        raise ValueError(
            f"fresh test 2 manifest {digest[:16]}... differs from the pinned "
            f"{FRESH_TEST_2_MANIFEST_SHA256[:16]}..."
        )
    reference = splits or exposed_splits
    near = near_duplicate_pairs(
        {"train": reference["train"], "validation": reference["validation"], "fresh_test_2": records},
        evaluated=("fresh_test_2",),
    )
    return {
        "records": records,
        "fresh_test_1": fresh_1,
        "manifest": {
            "source": f"test.csv at {CORPUS_RELEASE} (sha256 {CORPUS_FILES['test'][2]})",
            "seed": FRESH_TEST_2_SEED,
            "per_intent": FRESH_TEST_PER_INTENT,
            "n": len(records),
            "clean_pool": len(clean),
            "excluded": len(excluded & {query_key(r["query"]) for r in pool}),
            "near_duplicates_vs_train_validation": {
                "threshold_jaccard": NEAR_DUPLICATE_JACCARD,
                "in_fresh_test_2": near["near_duplicates"]["fresh_test_2"],
                "examples": near["examples"],
                "reported_not_removed": True,
            },
            "manifest_sha256": digest,
            "fresh_test_1_label": FRESH_TEST_1_LABEL,
        },
    }
