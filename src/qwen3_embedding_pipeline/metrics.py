"""Retrieval metrics over a query–positive dataset and a lexical baseline.

Every query is ranked against the dataset's document set (its sorted unique positives) by a similarity the
caller supplies — the embedder's cosine, or bag-of-words overlap for the baseline — and the rank of the
query's own positive is read: **recall@1**, **recall@5** and **recall@10** (the positive is within the first
*k*), and **MRR** (mean of 1 / rank). Ties are resolved pessimistically (a tied document counts as ranked
above the positive), so a similarity that cannot separate documents scores as badly as it deserves. The
**random floor** is what a uniformly random ranking over the document set achieves in expectation; the
**lexical baseline** ranks documents by the Jaccard overlap of lower-cased alphanumeric tokens with the
query — what a system with no model at all gets from shared words.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

RECALL_AT = (1, 5, 10)
METRIC_DEFINITIONS = {
    "recall@k": (
        "fraction of queries whose positive document is ranked within the first k of the document set; "
        "ties count against the positive"
    ),
    "mrr": "mean over queries of 1 / rank of the positive document",
    "document_set": "the sorted unique positive (and explicit negative) documents of the scored dataset",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def rank_of_positive(scores: Sequence[float], positive_index: int) -> int:
    """1-based rank of `positive_index` under descending `scores`; ties are ranked above the positive."""
    if not 0 <= positive_index < len(scores):
        raise ValueError("positive_index is outside the document set")
    target = float(scores[positive_index])
    return 1 + sum(1 for i, s in enumerate(scores) if i != positive_index and float(s) >= target)


def retrieval_metrics(ranks: Sequence[int], n_documents: int) -> dict[str, Any]:
    """Aggregate 1-based ranks of each query's positive into recall@k and MRR."""
    if not ranks:
        raise ValueError("no queries to score")
    if n_documents < 1:
        raise ValueError("n_documents must be positive")
    if any(not isinstance(r, int) or r < 1 or r > n_documents for r in ranks):
        raise ValueError("every rank must be an int in 1..n_documents")
    out: dict[str, Any] = {"n_queries": len(ranks), "n_documents": n_documents}
    for k in RECALL_AT:
        out[f"recall@{k}"] = sum(1 for r in ranks if r <= k) / len(ranks)
    out["mrr"] = sum(1.0 / r for r in ranks) / len(ranks)
    out["median_rank"] = sorted(ranks)[len(ranks) // 2]
    out["definitions"] = dict(METRIC_DEFINITIONS)
    return out


def random_floor(n_documents: int) -> dict[str, Any]:
    """Expected metrics of a uniformly random ranking over `n_documents` documents."""
    if n_documents < 1:
        raise ValueError("n_documents must be positive")
    out: dict[str, Any] = {"n_documents": n_documents}
    for k in RECALL_AT:
        out[f"recall@{k}"] = min(k, n_documents) / n_documents
    out["mrr"] = sum(1.0 / r for r in range(1, n_documents + 1)) / n_documents
    out["baseline"] = "uniformly random ranking of the document set (expected values)"
    return out


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def lexical_baseline(records: Sequence[Mapping[str, Any]], documents: Sequence[str]) -> dict[str, Any]:
    """Rank the documents for every query by Jaccard token overlap — the no-model floor."""
    index = {doc: i for i, doc in enumerate(documents)}
    ranks = []
    for record in records:
        if record["positive"] not in index:
            raise ValueError(f"positive {record['positive']!r} is not in the document set")
        scores = [jaccard(record["query"], doc) for doc in documents]
        ranks.append(rank_of_positive(scores, index[record["positive"]]))
    result = retrieval_metrics(ranks, len(documents))
    result["baseline"] = "Jaccard overlap of lower-cased alphanumeric tokens between query and document"
    return result


def paired_rank_comparison(
    ranks_a: Sequence[int],
    ranks_b: Sequence[int],
    *,
    n_resamples: int = 2_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Paired comparison of two systems on the same queries: the per-query reciprocal-rank difference (b -
    a), its mean (the MRR difference), how many queries improved or got worse, and a paired bootstrap
    percentile interval for the MRR difference (queries resampled with replacement, `n_resamples` times,
    seeded)."""
    if len(ranks_a) != len(ranks_b) or not ranks_a:
        raise ValueError("paired comparison needs two equal-length, non-empty rank lists")
    diffs = [1.0 / b - 1.0 / a for a, b in zip(ranks_a, ranks_b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    lo = means[int((1 - level) / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 + level) / 2 * n_resamples))]
    return {
        "n_queries": n,
        "delta_mrr": sum(diffs) / n,
        "ci": [lo, hi],
        "ci_level": level,
        "n_resamples": n_resamples,
        "seed": seed,
        "n_improved": sum(d > 0 for d in diffs),
        "n_worse": sum(d < 0 for d in diffs),
        "n_unchanged": sum(d == 0 for d in diffs),
        "method": "paired bootstrap over queries of the reciprocal-rank difference (percentile interval)",
    }


def paired_cluster_comparison(
    ranks_a: Sequence[int],
    ranks_b: Sequence[int],
    clusters: Sequence[str],
    *,
    n_resamples: int = 2_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, Any]:
    """The cluster (intent-level) version of `paired_rank_comparison`: whole clusters are resampled with
    replacement, keeping every query of a drawn cluster, so queries sharing an intent document are not treated
    as independent. A secondary analysis next to the query-level interval, not a replacement for it."""
    if not (len(ranks_a) == len(ranks_b) == len(clusters)) or not ranks_a:
        raise ValueError("cluster comparison needs equal-length, non-empty rank and cluster lists")
    by_cluster: dict[str, list[float]] = {}
    for a, b, c in zip(ranks_a, ranks_b, clusters, strict=True):
        by_cluster.setdefault(str(c), []).append(1.0 / b - 1.0 / a)
    groups = [by_cluster[k] for k in sorted(by_cluster)]
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        drawn = [groups[rng.randrange(len(groups))] for _ in groups]
        means.append(sum(sum(g) for g in drawn) / sum(len(g) for g in drawn))
    means.sort()
    lo = means[int((1 - level) / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 + level) / 2 * n_resamples))]
    diffs = [d for g in groups for d in g]
    return {
        "n_queries": len(diffs),
        "n_clusters": len(groups),
        "delta_mrr": sum(diffs) / len(diffs),
        "ci": [lo, hi],
        "ci_level": level,
        "n_resamples": n_resamples,
        "seed": seed,
        "method": (
            "paired cluster bootstrap over intents (whole intents resampled, percentile interval); secondary"
        ),
    }
