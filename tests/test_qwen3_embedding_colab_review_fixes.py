"""Regression tests for the Notebook Review Framework v1 findings on qwen3_embedding_colab.ipynb (QEM-M2, QEM-M3,
QEM-M4, QEM-m2, QEM-m3, QEM-m4, QEM-S1). QEM-M1 and QEM-m1 are covered by tests/test_sweep_fixes.py (SWP-R, SWP-B)
and tests/test_worker_colab_stubs.py.

Only CI's dependencies are used: the notebook's own cell sources run against the carried modules (no model) or
stand-ins. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import os
import types
from itertools import combinations
from pathlib import Path

import pytest

from qwen3_embedding_pipeline import pipeline as pl
from qwen3_embedding_pipeline import samples

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "qwen3_embedding_colab.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cell(notebook: dict, marker: str) -> str:
    found = [_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and marker in _source(c)]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _pairs(n: int, k: int) -> list[dict]:
    return [{"id": f"r{i:03d}", "query": f"question number {i} about topic {i % k}", "positive": f"topic {i % k}"} for i in range(n)]


# --- QEM-M3: small BYOD datasets reach adaptation; a too-small one is refused with its own size -------------------


@pytest.mark.parametrize(("n", "k"), [(12, 2), (20, 2), (20, 5), (49, 8)])
def test_qem_m3_small_byod_splits_validate_for_every_seed(n, k):
    for seed in range(60):
        splits = samples.split_dataset(_pairs(n, k), seed=seed)
        samples.validate_dataset(splits["train"])
        for name in ("validation", "test"):
            manifest = samples.validate_dataset(splits[name], min_records=1)
            assert manifest["n_documents"] >= samples.MIN_DOCUMENTS
        assert samples.check_split_disjoint(splits) == {name: len(part) for name, part in splits.items()}


@pytest.mark.parametrize("n", [3, 8, 11])
def test_qem_m3_too_small_names_the_dataset_size_and_the_minimum(n):
    assert samples.min_split_records() == 12
    with pytest.raises(ValueError, match=rf"the dataset has {n} distinct queries \(of {n} records\).*at least 12 distinct queries are required"):
        samples.split_dataset(_pairs(n, 2), seed=0)


def test_qem_m3_section_4_byod_path_reaches_the_split_with_20_records(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "pairs.csv"
    with data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "query", "positive"])
        writer.writeheader()
        writer.writerows(_pairs(20, 2))
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", f"BYOD_PATH = {str(data)!r}")
    namespace = {name: getattr(samples, name) for name in dir(samples) if not name.startswith("__")}
    namespace.update(os=os, Path=Path)
    exec(compile(source, "<section 4 byod>", "exec"), namespace)
    assert namespace["disjoint"] == {"test": 4, "validation": 3, "train": 13}
    assert namespace["dataset_manifests"]["validation"]["n_records"] == 3
    assert namespace["data_source"] == "BYOD (pairs.csv)" and (tmp_path / "outputs" / "qwen3_embedding_train.csv").is_file()
    assert set(namespace["near_duplicates"]["near_duplicates"]) == {"validation", "test"}


def test_qem_m3_prerequisites_and_troubleshooting_state_the_true_minimum(notebook):
    md = _markdown(notebook)
    assert "a BYOD dataset needs 12..20,000 distinct queries" in md
    assert "at least 12 distinct queries (8 stay for training)" in md
    assert "a dataset needs 8..20,000 records" not in md


def _section_6_tail(notebook: dict) -> str:
    source = _cell(notebook, "frozen_vs_floor = ")
    return source[source.index("# Contract integrity: both systems ranked the same document set.") :]


def _m(r1: float, mrr: float) -> dict:
    return {"recall@1": r1, "recall@5": r1, "recall@10": r1, "mrr": mrr, "median_rank": 1, "n_documents": 77}


def test_qem_m3_sample_path_still_stops_on_a_broken_frozen_model(notebook):
    namespace = {"frozen_test": _m(0.01, 0.05), "baseline_lexical": _m(0.3, 0.4), "floor": _m(0.013, 0.064), "USE_BYOD": False}
    with pytest.raises(RuntimeError, match="frozen model must rank above the random floor"):
        exec(compile(_section_6_tail(notebook), "<section 6>", "exec"), namespace)
    namespace.update(USE_BYOD=True)
    exec(compile(_section_6_tail(notebook), "<section 6>", "exec"), namespace)
    assert namespace["frozen_vs_floor"] == "not above"


def _section_8_namespace(adapter, adapted_mrr, use_byod, default_settings):
    return {
        "json": json, "pipe": types.SimpleNamespace(adapter=adapter, evaluate=lambda records, **kw: _m(0.7, adapted_mrr)), "test_records": [], "val_records": [],
        "documents": lambda records: [], "document_set": ["d"] * 77, "INSTRUCTION": "i", "floor": _m(0.013, 0.064), "baseline_lexical": _m(0.3, 0.4),
        "frozen_test": _m(0.63, 0.741), "frozen_vs_floor": "above", "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k", "data_source": "stand-in",
        "dataset_manifests": {}, "disjoint": {}, "adapt_result": {"history": [], "best_epoch": 0}, "adapt_seconds": 0.0, "near_duplicates": {"near_duplicates": {"test": 9}},
        "USE_BYOD": use_byod, "DEFAULT_SETTINGS": default_settings,
    }


def test_qem_m3_section_8_holds_only_the_default_sample_run_to_an_improvement(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    source = _cell(notebook, "delta_mrr = ")
    with pytest.raises(RuntimeError, match="adapted MRR must be above the frozen MRR"):
        exec(compile(source, "<section 8>", "exec"), _section_8_namespace({}, 0.6, False, True))
    for use_byod, default_settings in ((True, True), (False, False)):
        namespace = _section_8_namespace({}, 0.6, use_byod, default_settings)
        exec(compile(source, "<section 8>", "exec"), namespace)
        assert namespace["comparison"]["verdict"]["adapted_vs_frozen_mrr"] == "worse"
    report = json.loads((tmp_path / "outputs" / "qwen3_embedding_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["near_duplicates"] == {"near_duplicates": {"test": 9}}


# --- QEM-M2: measurements of the frozen model always use the pinned base; adapted cells refuse the base ----------


class _Pipe:
    def __init__(self, adapted: bool):
        self.adapter = {"trainable_names": ["layers.27.w"]} if adapted else None
        self.calls: list[str] = []

    def restore_base(self):
        self.calls.append("restore_base")
        was = self.adapter is not None
        self.adapter = None
        return ["layers.27.w"] if was else []

    def embed(self, texts, kind="document", instruction=None):
        self.calls.append(f"embed:{kind}:adapted={self.adapter is not None}")
        raise _Stop


class _Stop(Exception):
    pass


def test_qem_m2_section_5_puts_the_pinned_base_back_before_embedding(notebook, tmp_path, monkeypatch, capsys):
    source = _cell(notebook, "probe_records = test_records[:3]")
    assert source.index("pipe.restore_base()") < source.index("pipe.embed(")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = _Pipe(adapted=True)
    namespace = {name: getattr(pl, name) for name in dir(pl) if not name.startswith("__")}
    namespace.update(pipe=pipe, test_records=_pairs(3, 3), document_set=[f"doc {i}" for i in range(77)], INSTRUCTION="i", json=json)
    with pytest.raises(_Stop):
        exec(compile(source, "<section 5>", "exec"), namespace)
    assert pipe.calls[0] == "restore_base" and pipe.calls[1] == "embed:document:adapted=False"
    assert len(json.loads((tmp_path / "outputs" / "qwen3_embedding_input_manifest.json").read_text(encoding="utf-8"))["inputs"]) == 77
    assert "restored_pinned_base" in capsys.readouterr().out


def test_qem_m2_section_6_restores_before_the_frozen_evaluation(notebook):
    source = _cell(notebook, "frozen_vs_floor = ")
    assert source.index("pipe.restore_base()") < source.index("frozen_test = pipe.evaluate(")


@pytest.mark.parametrize("marker", ["delta_mrr = ", "pipe.save_artifact("])
def test_qem_m2_sections_8_and_9_refuse_the_pinned_base(notebook, marker):
    source = _cell(notebook, marker)
    guard = source[source.index("if pipe.adapter is None:") :]
    guard = guard[: guard.index("\n", guard.index("raise RuntimeError")) + 1]
    with pytest.raises(RuntimeError, match="holds the pinned base"):
        exec(compile(guard, "<guard>", "exec"), {"pipe": types.SimpleNamespace(adapter=None)})
    exec(compile(guard, "<guard>", "exec"), {"pipe": types.SimpleNamespace(adapter={})})


def test_qem_m2_opening_and_activity_say_which_cells_to_rerun(notebook):
    md = _markdown(notebook)
    assert "Sections 5 and 6 put the pinned base back before they measure" in md
    assert "Run Section 7, then Section 8." in md
    assert "Section 8 or 9 says the pipeline holds the pinned base" in md


# --- QEM-M4: a Predict -> Change -> Run -> Observe -> Explain activity -------------------------------------------


def test_qem_m4_activity_follows_the_five_steps(notebook):
    md = _markdown(notebook)
    start = md.index("## Activity: how much does the second trainable layer buy?")
    steps = [md.index(f"{i}. **{name}.**", start) for i, name in enumerate(("Predict", "Change", "Run", "Observe", "Explain"), 1)]
    assert steps == sorted(steps)
    assert "<details><summary>Check your reasoning</summary>" in md[steps[-1] :]
    assert "To put the notebook back to the recorded state" in md


# --- QEM-m2: leakage key and near-duplicate report --------------------------------------------------------------


def test_qem_m2_minor_split_check_ignores_case_punctuation_and_spacing():
    assert samples.query_key("Has my  top-up been cancelled?") == samples.query_key("has my top up been cancelled")
    splits = {"train": [{"query": "My top-up has been cancelled."}], "test": [{"query": "my top up has been cancelled"}]}
    with pytest.raises(ValueError, match="appears in both train and test"):
        samples.check_split_disjoint(splits)


def test_qem_m2_minor_near_duplicates_match_a_brute_force_scan():
    texts = [
        "Has my top-up been cancelled?", "My top-up has been cancelled.", "How can I top up my account with a cheque?",
        "How can I top up my account with a card?", "Where is my new card", "I want to close my account",
        "why was my card declined at the shop", "my card was declined at the shop why",
    ]
    splits = {"train": [{"query": t} for t in texts[0::2]], "test": [{"query": t} for t in texts[1::2]], "validation": []}
    report = samples.near_duplicate_pairs(splits)

    def jaccard(a, b):
        ta, tb = samples._tokens(a), samples._tokens(b)
        return len(ta & tb) / len(ta | tb)

    brute = sum(any(jaccard(q["query"], o["query"]) >= 0.8 for o in splits["train"]) for q in splits["test"])
    assert report["near_duplicates"] == {"validation": 0, "test": brute} and brute == 3
    assert all(e["jaccard"] >= 0.8 for e in report["examples"])
    assert all(jaccard(a, b) < 0.8 for a, b in combinations(texts[4:6], 2))


def test_qem_m2_minor_section_4_prints_the_count_and_the_prose_is_honest(notebook):
    source = _cell(notebook, "USE_BYOD = False")
    assert "near_duplicates = near_duplicate_pairs(splits)" in source
    md = _markdown(notebook)
    assert "9 of the 385 test queries at token Jaccard ≥ 0.8" in md
    assert "without leakage" not in md


# --- QEM-m3: the input manifest covers every document ------------------------------------------------------------


def test_qem_m3_minor_input_manifest_lists_all_77_documents(notebook):
    source = _cell(notebook, "probe_records = test_records[:3]")
    block = source[source.index("document_names = ") : source.index("input_manifest['query_manifest']")]
    namespace = {"validate_inputs": pl.validate_inputs, "MAX_BATCH": pl.MAX_BATCH, "document_set": [f"intent phrase {i}" for i in range(77)]}
    exec(compile(block, "<section 5 manifest>", "exec"), namespace)
    manifest = namespace["input_manifest"]
    assert pl.MAX_BATCH == 64 and manifest["batches"] == 2
    assert [entry["id"] for entry in manifest["inputs"]] == [f"doc-{i:02d}" for i in range(77)]
    assert "an input manifest for all 77 documents" in _markdown(notebook)


# --- QEM-m4: device-labelled figures; QEM-S1: spec 2.2 ------------------------------------------------------------


def test_qem_m4_section_7_labels_the_device_of_every_quoted_figure(notebook):
    md = _markdown(notebook)
    assert "The CPU float32 build record's sweep on this sample" in md
    assert "the recorded Kaggle Tesla T4 run of the default reached 76.6 %" in md


def test_qem_s1_notebook_declares_spec_2_2(notebook):
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    md = _markdown(notebook)
    assert "DIMER Notebook Specification 2.2" in md and "DIMER Notebook Specification 2.0" not in md
