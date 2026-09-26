"""Exercise real notebook optional-path orchestration with deterministic model doubles.

These tests do not establish model quality or hosted-runtime qualification.
"""
import ast
import csv
import gc
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

NOTEBOOK = (Path(__file__).resolve().parents[1] / 'tutorials'
            / 'DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb')


def cell(index):
    return ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][index]['source'])


def setup(tmp_path, labels):
    docs = tmp_path / 'documents.csv'
    queries = tmp_path / 'queries.csv'
    docs.write_text('doc_id,text\na,alpha\nb,beta\n', encoding='utf-8')
    with queries.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['query_id', 'query', 'gold_doc_id'])
        writer.writeheader()
        writer.writerows({'query_id': f'q{i}', 'query': 'alpha', 'gold_doc_id': value}
                        for i, value in enumerate(labels))
    live = {'embedding': 0, 'reranker': 0}

    class Embedding:
        def __init__(self, root):
            assert live['reranker'] == 0, 'overlapping model residency'
            live['embedding'] += 1

        def embed_all(self, texts, kind, instruction):
            if kind == 'document':
                return np.eye(2), [1, 1]
            return np.tile([1., 0.], (len(texts), 1)), [1] * len(texts)

        def __del__(self):
            live['embedding'] -= 1

    class Reranker:
        def __init__(self, root):
            assert live['embedding'] == 0, 'overlapping model residency'
            live['reranker'] += 1

        def score_all(self, pairs, instruction):
            return [1. if d == 'alpha' else 0. for _, d in pairs], [1] * len(pairs)

        def __del__(self):
            live['reranker'] -= 1

    ns = dict(Path=Path, csv=csv, gc=gc, hashlib=hashlib, json=json, np=np,
              statistics=statistics, torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
              USE_BYOD=True, DOCUMENTS_PATH=str(docs), QUERIES_PATH=str(queries),
              MAX_TEXT_CHARS=100000, RERANK_K=6, EMBED_DIR='unused', RERANK_DIR='unused',
              EMBEDDING_INSTRUCTION='embed', RERANK_INSTRUCTION='rerank',
              EMBED_MODEL_REVISION='embedding-revision', RERANK_MODEL_REVISION='reranker-revision',
              RUNTIME={'test_double': True}, OUTPUT_DIR=str(tmp_path / 'outputs'),
              EmbeddingRuntime=Embedding, RerankerRuntime=Reranker, reranker=Reranker('unused'))
    tree = ast.parse(cell(14))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'metrics_from_ranks']
    exec(compile(tree, 'notebook-metrics', 'exec'), ns)
    return ns, live


@pytest.mark.parametrize('labels', [['a', 'b'], ['', '']])
def test_byod_exports_labelled_or_unlabelled_results_with_one_model_resident(tmp_path, labels):
    ns, live = setup(tmp_path, labels)
    canonical = tmp_path / 'outputs/metrics.json'
    canonical.parent.mkdir()
    canonical.write_text('canonical marker')
    exec(cell(38), ns)
    result = json.loads((canonical.parent / 'byod_results.json').read_text())
    assert result['labelled'] == bool(labels[0])
    assert result['evaluation_verdict'] == ('measured' if labels[0] else 'not-measurable')
    assert result['provenance']['effective_rerank_k'] == 2
    assert result['provenance']['documents_sha256'] == hashlib.sha256(
        Path(ns['DOCUMENTS_PATH']).read_bytes()).hexdigest()
    assert len(list(csv.DictReader((canonical.parent / 'byod_rankings.csv').open()))) == 4
    if labels[0]:
        assert result['pipeline_recall@1'] == 0.5
    else:
        assert 'pipeline_recall@1' not in result
    assert live == {'embedding': 0, 'reranker': 1}
    assert canonical.read_text() == 'canonical marker'


@pytest.mark.parametrize(('labels', 'message'), [
    (['a', ''], 'every query'),
    (['unknown'], 'unknown gold_doc_id'),
])
def test_byod_invalid_labels_fail_before_model_reload(tmp_path, labels, message):
    ns, live = setup(tmp_path, labels)
    original = ns['reranker']
    with pytest.raises(ValueError, match=message):
        exec(cell(38), ns)
    assert ns['reranker'] is original
    assert live == {'embedding': 0, 'reranker': 1}
    assert not (tmp_path / 'outputs').exists()


def test_depth_sweep_preserves_canonical_results():
    class Reranker:
        def score_all(self, pairs, instruction):
            return [1. if doc == 'gold' else 0. for _, doc in pairs], []

    ranking = [{'doc_id': 'a', 'is_gold': False}, {'doc_id': 'b', 'is_gold': True}]
    canonical = {'recall@1': 0.0}
    ns = dict(RUN_K_SWEEP=True, K_SWEEP_VALUES=[1, 2], queries=[{'query_id': 'q', 'query': 'gold'}],
              retrieval_rankings=[ranking], documents=[{'text': 'wrong'}, {'text': 'gold'}],
              doc_id_to_index={'a': 0, 'b': 1}, time=time, defaultdict=defaultdict,
              reranker=Reranker(), RERANK_INSTRUCTION='rank', fmt_pct=lambda x: f'{x:.1%}',
              RERANK_K=6, pipeline_metrics=canonical)
    exec(cell(34), ns)
    assert [row['recall@1'] for row in ns['k_sweep_results']] == [0., 1.]
    assert [row['pairs'] for row in ns['k_sweep_results']] == [1, 2]
    assert ns['RERANK_K'] == 6 and ns['pipeline_metrics'] is canonical
    assert canonical == {'recall@1': 0.0}
    assert ranking == [{'doc_id': 'a', 'is_gold': False}, {'doc_id': 'b', 'is_gold': True}]

@pytest.mark.parametrize('stage', ['load', 'embed'])
def test_failed_embedding_restores_reranker_and_allows_retry(tmp_path, stage):
    ns, live = setup(tmp_path, ['a'])
    original = ns['EmbeddingRuntime']
    if stage == 'load':
        def fail(root):
            raise RuntimeError('transient embedding failure')
        ns['EmbeddingRuntime'] = fail
    else:
        class FailingEmbedding(original):
            def embed_all(self, *args):
                raise RuntimeError('transient embedding failure')
        ns['EmbeddingRuntime'] = FailingEmbedding
    with pytest.raises(RuntimeError, match='transient embedding failure'):
        exec(cell(38), ns)
    gc.collect()
    assert live == {'embedding': 0, 'reranker': 1}
    assert 'byod_embedder' not in ns
    ns['EmbeddingRuntime'] = original
    exec(cell(38), ns)
    assert ns['byod_result']['pipeline_recall@1'] == 1.0
    assert live == {'embedding': 0, 'reranker': 1}
