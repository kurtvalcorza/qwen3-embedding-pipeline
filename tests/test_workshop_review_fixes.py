"""Regression tests for the 2026-09-28 notebook review findings (QSR-01 .. QSR-10).

The notebook cells are executed with synthetic fixtures and deterministic model doubles. These tests check
metric arithmetic, output validation and learner-facing views; they do not establish model quality or
hosted-runtime qualification.
"""
import json
import math
import statistics
import time
from collections import defaultdict

import numpy as np
import pytest

from test_workshop_optional_paths import cell, load_shared_helpers, setup


def helpers():
    return load_shared_helpers({})


# QSR-02 - one tie policy for every system, with the conservative figure as a named diagnostic.
def lexical_fixture():
    documents = [
        {"doc_id": "d0", "text": "card payment", "intent": "card_payment"},
        {"doc_id": "d1", "text": "card transfer", "intent": "card_transfer"},
        {"doc_id": "d2", "text": "exchange rate", "intent": "exchange_rate"},
    ]
    queries = [
        {"query_id": "q0", "query": "card", "gold_doc_id": "d1"},  # tied with d0, displayed second
        {"query_id": "q1", "query": "card", "gold_doc_id": "d0"},  # tied with d1, displayed first
        {"query_id": "q2", "query": "exchange rate", "gold_doc_id": "d2"},  # unique best score
        {"query_id": "q3", "query": "zzz", "gold_doc_id": "d2"},  # all scores equal (zero)
    ]
    ns = {"np": np, "documents": documents, "queries": queries, "RERANK_K": 6,
          "doc_id_to_index": {d["doc_id"]: i for i, d in enumerate(documents)}}
    exec(cell(14), ns)
    return ns


def test_lexical_metric_ranks_are_the_displayed_ranks_under_ties():
    ns = lexical_fixture()
    displayed = {row["query_id"]: row["rank"] for row in ns["lexical_rows"] if row["is_gold"]}
    assert [displayed[q["query_id"]] for q in ns["queries"]] == ns["lexical_ranks"] == [2, 1, 1, 3]
    assert ns["lexical_metrics"]["recall@1"] == 0.5
    assert ns["lexical_metrics"]["mrr"] == pytest.approx((1 / 2 + 1 + 1 + 1 / 3) / 4)
    # The conservative policy is still reported, under its own name.
    assert ns["lexical_pessimistic_ranks"] == [2, 2, 1, 3]
    assert ns["lexical_pessimistic_metrics"]["recall@1"] == 0.25
    assert ns["lexical_tie_diagnostic"] == {"queries_with_gold_score_tied": 3, "fraction": 0.75}


def test_dense_and_lexical_orderings_agree_for_the_same_scores():
    h = helpers()
    for scores in ([0.5, 0.5, 0.0], [0.1, 0.9, 0.4], [0.0, 0.0, 0.0]):
        dense = np.argsort(-np.asarray(scores), kind="stable").tolist()
        assert h["stable_order"](scores) == dense


# QSR-03 - fixed cutoffs are literal; deeper cutoffs than the composed list are not measurable.
@pytest.mark.parametrize("depth", [2, 3, 6, 10, 32])
def test_composed_metrics_use_literal_cutoffs(depth):
    h = helpers()
    raw = [1, 2, 3, 5, 8, 12, 40]
    ranks = [r if r <= depth else None for r in raw]
    result = h["composed_metrics"](ranks, depth, ks=(1, 3, 6))
    for k in sorted({1, 3, 6, depth}):
        expected = None if k > depth else sum(r is not None and r <= k for r in ranks) / len(ranks)
        assert result[f"recall@{k}"] == expected, k
    assert result["mrr"] == pytest.approx(sum(0.0 if r is None else 1 / r for r in ranks) / len(ranks))


def test_review_probe_changed_k_is_not_reported_under_r_at_6():
    result = helpers()["composed_metrics"]([8], 10, ks=(1, 3, 6))
    assert result["recall@6"] == 0.0 and result["recall@10"] == 1.0


def comparison_namespace(depth):
    h = helpers()
    ks = tuple(sorted({1, 3, 6, 10, depth}))
    final = [r if r <= depth else None for r in [1, 1, 3, 5, 8]]
    ns = dict(h)
    ns.update(
        RERANK_K=depth,
        queries=[{}] * 5,
        random_metrics=h["random_floor"](77, ks=ks),
        lexical_metrics=h["metrics_from_ranks"]([1, 3, 5, 9, 20], 77, ks=ks),
        lexical_pessimistic_metrics=h["metrics_from_ranks"]([2, 4, 5, 9, 20], 77, ks=ks),
        lexical_tie_diagnostic={"queries_with_gold_score_tied": 2, "fraction": 0.4},
        retriever_metrics=h["metrics_from_ranks"]([1, 2, 4, 7, 9], 77, ks=ks),
        pipeline_metrics=h["composed_metrics"](final, depth, ks=(1, 3, 6)),
    )
    return ns


@pytest.mark.parametrize("depth", [2, 6, 10])
def test_comparison_table_labels_every_column_by_its_literal_cutoff(depth, capsys):
    ns = comparison_namespace(depth)
    exec(cell(30), ns)
    out = capsys.readouterr().out
    header = out.splitlines()[0]
    pipeline = ns["summary_rows"][-1]
    assert pipeline["recall@6"] == ns["pipeline_metrics"]["recall@6"]
    if depth == 10:
        assert pipeline["recall@6"] == 0.8 and pipeline["recall@10"] == 1.0
        assert "R@10" in header
    if depth == 2:
        assert pipeline["recall@3"] is None and pipeline["recall@6"] is None
        assert "R@2" in header and "n/a" in out
    if depth == 6:
        assert header.count("R@6") == 1
    assert "ties against gold" in out


# QSR-05 - zero shortlist coverage yields an explicit not-applicable conditional record.
def evaluation_namespace(covered):
    queries = [{"query_id": f"q{i}"} for i in range(2)]
    shortlists, reranked = [], {}
    for i, q in enumerate(queries):
        gold_here = covered and i == 0
        candidates = [{"doc_id": "g" if gold_here and j == 1 else f"x{j}", "retrieval_rank": j + 1,
                       "is_gold": gold_here and j == 1} for j in range(3)]
        shortlists.append({"query_id": q["query_id"], "candidates": candidates})
        ordered = sorted(candidates, key=lambda c: not c["is_gold"])
        reranked[q["query_id"]] = [dict(c, reranked_rank=r + 1) for r, c in enumerate(ordered)]
    ns = helpers()
    ns.update(queries=queries, shortlists=shortlists, reranked_by_query=reranked, RERANK_K=3,
              shortlist_coverage=0.5 if covered else 0.0)
    return ns


def test_zero_coverage_conditional_metrics_are_not_applicable():
    ns = evaluation_namespace(covered=False)
    exec(cell(28), ns)
    conditional = ns["conditional_metrics"]
    assert conditional["applicable"] is False and conditional["n_queries"] == 0
    assert conditional["mrr"] is None and conditional["recall@1"] is None
    assert ns["pipeline_metrics"]["recall@1"] == 0.0 and ns["effects"]["unrecoverable"] == 2
    json.dumps(conditional)  # exportable


def test_covered_conditional_metrics_are_measured():
    ns = evaluation_namespace(covered=True)
    exec(cell(28), ns)
    assert ns["conditional_metrics"]["n_queries"] == 1
    assert ns["conditional_metrics"]["recall@1"] == 1.0
    assert ns["effects"]["helped"] == 1


# QSR-01 - learner views show the query, gold phrase and candidate phrases.
def views_namespace():
    documents = [{"doc_id": "d0", "text": "cash withdrawal"}, {"doc_id": "d1", "text": "replace stolen card"},
                 {"doc_id": "d2", "text": "exchange rate"}]
    queries = [{"query_id": "q0", "query": "my card was stolen", "gold_doc_id": "d1"},
               {"query_id": "q1", "query": "what rate do you use", "gold_doc_id": "d2"}]

    def row(qid, doc, rank, gold):
        return {"query_id": qid, "doc_id": doc, "retrieval_rank": rank, "retrieval_score": 0.9 - rank / 10,
                "is_gold": gold}

    rankings = [[row("q0", "d0", 1, False), row("q0", "d1", 2, True), row("q0", "d2", 3, False)],
                [row("q1", "d0", 1, False), row("q1", "d1", 2, False), row("q1", "d2", 3, True)]]
    shortlists = [{"query_id": q["query_id"], "query": q["query"], "gold_doc_id": q["gold_doc_id"],
                   "candidates": r[:2]} for q, r in zip(queries, rankings, strict=True)]
    reranked = {
        "q0": [dict(rankings[0][1], reranker_score=0.9, reranked_rank=1),
               dict(rankings[0][0], reranker_score=0.1, reranked_rank=2)],
        "q1": [dict(rankings[1][0], reranker_score=0.6, reranked_rank=1),
               dict(rankings[1][1], reranker_score=0.2, reranked_rank=2)],
    }
    return dict(documents=documents, queries=queries, doc_id_to_index={"d0": 0, "d1": 1, "d2": 2},
                shortlists=shortlists, retrieval_rankings=rankings, reranked_by_query=reranked,
                effect_by_query={"q0": "helped", "q1": "unrecoverable"}, RERANK_K=2)


def test_prediction_and_failure_views_show_candidate_and_gold_text(capsys):
    ns = views_namespace()
    exec(cell(22), ns)
    exercise = capsys.readouterr().out
    assert "cash withdrawal" in exercise and "replace stolen card" in exercise
    assert 'Gold:  d1  "replace stolen card"' in exercise
    exec(cell(32), ns)
    cases = capsys.readouterr().out
    assert "CASE: HELPED" in cases and "Gold rank: 2 \u2192 1" in cases
    assert '"exchange rate"' in cases  # gold phrase of the unrecoverable case
    assert "first-stage rank 3 of 3, outside top-2" in cases


# QSR-04 - shared output validation on the canonical, sweep and BYOD paths.
@pytest.mark.parametrize(("vectors", "message"), [
    (np.eye(3)[:2], "shape"),
    (np.array([[1.0, 0.0], [np.nan, 0.0]]), "non-finite"),
    (np.array([[1.0, 0.0], [0.5, 0.0]]), "unit norm"),
])
def test_embedding_checks_reject_invalid_matrices(vectors, message):
    with pytest.raises(RuntimeError, match=message):
        helpers()["check_embedding_matrix"](vectors, 2, 2, "fixture")


@pytest.mark.parametrize(("scores", "message"), [
    ([0.2], "1 scores for 2 pairs"),
    ([0.2, math.nan], "out-of-range"),
    ([0.2, 1.5], "out-of-range"),
    ([-0.1, 0.5], "out-of-range"),
])
def test_score_checks_reject_invalid_scores(scores, message):
    with pytest.raises(RuntimeError, match=message):
        helpers()["check_relevance_scores"](scores, 2, "fixture")


def test_permutation_and_rerank_group_checks():
    h = helpers()
    assert h["check_permutation"](np.array([2, 0, 1]), 3, "ok") == [2, 0, 1]
    with pytest.raises(RuntimeError, match="permutation"):
        h["check_permutation"]([0, 0, 1], 3, "fixture")
    rows = [{"doc_id": "a", "reranked_rank": 1}, {"doc_id": "b", "reranked_rank": 2}]
    h["check_reranked_group"](["b", "a"], rows, "ok")
    with pytest.raises(RuntimeError, match="unique shortlist"):
        h["check_reranked_group"](["a", "c"], rows, "fixture")


@pytest.mark.parametrize("fault", ["nan-vector", "missing-row", "short-scores", "out-of-range-score"])
def test_byod_rejects_invalid_model_outputs_before_export(tmp_path, fault):
    ns, live = setup(tmp_path, ["a", "b"])
    embedding, reranker = ns["EmbeddingRuntime"], type(ns["reranker"])

    class BadEmbedding(embedding):
        def embed_all(self, texts, kind, instruction):
            vectors, counts = super().embed_all(texts, kind, instruction)
            if kind == "document" and fault == "nan-vector":
                vectors = vectors.copy()
                vectors[0, 0] = np.nan
            if kind == "document" and fault == "missing-row":
                vectors, counts = vectors[:1], counts[:1]
            return vectors, counts

    class BadReranker(reranker):
        def score_all(self, pairs, instruction):
            scores, counts = super().score_all(pairs, instruction)
            if fault == "short-scores":
                return scores[:-1], counts[:-1]
            if fault == "out-of-range-score":
                return [1.5] + scores[1:], counts
            return scores, counts

    ns["EmbeddingRuntime"], ns["RerankerRuntime"] = BadEmbedding, BadReranker
    with pytest.raises(RuntimeError, match="BYOD"):
        exec(cell(38), ns)
    assert not (tmp_path / "outputs" / "byod_results.json").exists()
    assert "byod_embedder" not in ns


def test_depth_sweep_rejects_out_of_range_scores():
    class Reranker:
        def score_all(self, pairs, instruction):
            return [2.0 for _ in pairs], []

    ranking = [{"doc_id": "a", "is_gold": False}, {"doc_id": "b", "is_gold": True}]
    ns = dict(RUN_K_SWEEP=True, K_SWEEP_VALUES=[2], queries=[{"query_id": "q", "query": "gold"}],
              retrieval_rankings=[ranking], documents=[{"text": "wrong"}, {"text": "gold"}],
              doc_id_to_index={"a": 0, "b": 1}, time=time, defaultdict=defaultdict, reranker=Reranker(),
              RERANK_INSTRUCTION="rank", fmt_pct=lambda x: f"{x:.1%}", RERANK_K=6, statistics=statistics)
    load_shared_helpers(ns)
    with pytest.raises(RuntimeError, match="K-sweep K=2"):
        exec(cell(34), ns)


# QSR-06 - ambiguous or malformed CSV structure fails with file/line diagnostics before model load.
@pytest.mark.parametrize(("content", "message"), [
    ("doc_id,text,text\na,alpha,x\nb,beta,y\n", "duplicate column names \\['text'\\]"),
    ("doc_id,text\na,alpha,extra\nb,beta\n", "line 2: expected 2 fields"),
    ("doc_id,text\na,alpha\nb\n", "line 3: expected 2 fields"),
    ("doc_id,,text\na,1,alpha\nb,2,beta\n", "empty header name"),
    ("doc_id,text\na,alpha\na,beta\n", "repeated \\['a'\\]"),
    ("doc_id,text\na,alpha\nb,   \n", "line 3, column 'text'"),
    ("doc_id,text\n", "no data rows"),
])
def test_byod_csv_structure_errors_name_the_location(tmp_path, content, message):
    ns, live = setup(tmp_path, ["", ""])
    original = ns["reranker"]
    (tmp_path / "documents.csv").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        exec(cell(38), ns)
    assert ns["reranker"] is original
    assert live == {"embedding": 0, "reranker": 1}
    assert not (tmp_path / "outputs").exists()


def test_byod_accepts_quoted_commas_and_a_spreadsheet_bom(tmp_path):
    ns, _ = setup(tmp_path, ["a", "b"])
    (tmp_path / "documents.csv").write_bytes('\ufeffdoc_id,text\na,alpha\nb,"beta, gamma"\n'.encode())
    exec(cell(38), ns)
    assert ns["byod_result"]["documents"] == 2


# QSR-07 / QSR-08 - task instructions are explicit and truncation is reported per input.
def test_byod_uses_and_records_user_instructions_and_truncation(tmp_path):
    ns, _ = setup(tmp_path, ["a", "b"])
    seen = {}
    embedding, reranker = ns["EmbeddingRuntime"], type(ns["reranker"])

    class RecordingEmbedding(embedding):
        def embed_all(self, texts, kind, instruction):
            seen[f"embed-{kind}"] = instruction
            return super().embed_all(texts, kind, instruction)

        def token_lengths(self, texts, kind, instruction):
            if kind == "document":
                return [9000 if t == "alpha" else 1 for t in texts]
            return [1] * len(texts)

    class RecordingReranker(reranker):
        def score_all(self, pairs, instruction):
            seen["rerank"] = instruction
            return super().score_all(pairs, instruction)

        def token_lengths(self, pairs, instruction):
            return [2 if d == "alpha" else 1 for _, d in pairs]

    ns.update(EmbeddingRuntime=RecordingEmbedding, RerankerRuntime=RecordingReranker,
              BYOD_EMBEDDING_INSTRUCTION="Given a question, retrieve the passage that answers it",
              BYOD_RERANK_INSTRUCTION="Judge whether the passage answers the question")
    exec(cell(38), ns)
    result = json.loads((tmp_path / "outputs" / "byod_results.json").read_text())
    assert seen == {"embed-document": ns["BYOD_EMBEDDING_INSTRUCTION"],
                    "embed-query": ns["BYOD_EMBEDDING_INSTRUCTION"], "rerank": ns["BYOD_RERANK_INSTRUCTION"]}
    assert result["provenance"]["instruction_source"] == {"embedding": "user-supplied",
                                                          "reranker": "user-supplied"}
    assert result["provenance"]["embedding_instruction"] == ns["BYOD_EMBEDDING_INSTRUCTION"]
    truncation = result["truncation"]
    assert truncation["documents"]["truncated"] == 1
    assert truncation["documents"]["examples"] == [{"id": "a", "tokens_before": 9000, "tokens_kept": 1}]
    assert truncation["queries"]["truncated"] == 0
    assert truncation["reranker_pairs"]["truncated"] == 2  # each query's pair with document "alpha"


def test_byod_default_instructions_are_labelled(tmp_path, capsys):
    ns, _ = setup(tmp_path, ["", ""])
    exec(cell(38), ns)
    assert "banking-intent instruction" in capsys.readouterr().out
    source = ns["byod_result"]["provenance"]["instruction_source"]
    assert source == {"embedding": "default banking-intent task", "reranker": "default banking-intent task"}


# QSR-09 / QSR-10 - static contract checks for the reranker call and the learner-facing wording.
def test_reranker_scores_only_the_final_position():
    source = cell(24)
    assert "logits_to_keep=1" in source and "use_cache=False" in source
    assert "def token_lengths" in source and "def token_lengths" in cell(16)


def test_exports_lookups_identity_and_separate_byod_recap():
    export, summary = cell(36), cell(41)
    for name in ("documents_lookup.csv", "queries_lookup.csv"):
        assert name in export and name in summary and name in cell(35)
    assert '"notebook": NOTEBOOK_IDENTITY' in export
    assert "lexical_ties_against_gold_diagnostic" in export
    assert "not comparable with the Banking77 numbers" in summary


def test_setup_and_troubleshooting_wording():
    assert "expected on the first run" not in cell(5)
    assert "normally continues without a restart" in cell(5)
    assert "default execution tier" not in cell(45)
    assert "BYOD_EMBEDDING_INSTRUCTION" in cell(4) and "BYOD_RERANK_INSTRUCTION" in cell(37)


# Length-aware batching (found in local long-input BYOD verification): bounded attention size per batch.
def test_short_inputs_keep_the_fixed_size_batches():
    h = helpers()
    for n, max_items, length in ((77, 64, 40), (154, 64, 130), (924, 32, 200), (70, 32, 1448)):
        batches = h["attention_budget_batches"]([length] * n, max_items, 8192 ** 2)
        assert batches == [(s, min(s + max_items, n)) for s in range(0, n, max_items)]


def test_long_inputs_get_bounded_batches():
    h = helpers()
    lengths = [100] * 5 + [8192] + [100] * 40 + [4000] * 6
    batches = h["attention_budget_batches"](lengths, 32, 8192 ** 2)
    assert [e for _, e in batches][-1] == len(lengths)
    assert all(s < e for s, e in batches)
    assert all(b[1] == c[0] for b, c in zip(batches, batches[1:], strict=False))
    for s, e in batches:
        assert e - s <= 32
        assert (e - s) * max(lengths[s:e]) ** 2 <= 8192 ** 2 or e - s == 1
    assert (5, 6) in batches  # the full-length input runs alone
    assert h["attention_budget_batches"]([], 32, 8192 ** 2) == []


def test_runtimes_batch_by_attention_budget():
    assert "attention_budget_batches(lengths, EMBED_MAX_BATCH, EMBED_BATCH_ATTENTION)" in cell(16)
    assert "attention_budget_batches(lengths, RERANK_MAX_PAIRS, RERANK_BATCH_ATTENTION)" in cell(24)
