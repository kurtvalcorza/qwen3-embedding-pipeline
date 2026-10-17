"""Regression tests for the row-72 review of qwen3_embedding_colab.ipynb (2026-10-17): R72-M1 (validation grid and
fresh test), R72-M2 (DAT9 disclosure and the CLINC150 transfer check), R72-M3 (score-once wiring, SCORE_ONCE_RULE
item 7), R72-m2 (provenance stamp) and R72-m3 (spec 2.5, DAT20).

Only CI's dependencies are used: the notebook's own cell sources run against the carried modules with stand-in
models and synthetic corpora. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import types
from pathlib import Path

import pytest

from qwen3_embedding_pipeline import metrics, samples

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "qwen3_embedding_colab.ipynb"
WORKSHOP_SOURCE = ROOT / "tools" / "semantic_search_reranking_workshop_source.py"


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


def _all_text(notebook: dict) -> str:
    return "\n".join(_source(c) for c in notebook["cells"])


# ---- stand-ins for Section 8 ------------------------------------------------------------------------------------


def _m(r1: float, mrr: float, n: int = 3) -> dict:
    return {"recall@1": r1, "recall@5": r1, "recall@10": r1, "mrr": mrr, "median_rank": 1, "n_documents": 77, "n_queries": n, "ranks": [1] * n}


class _Pipe:
    """Section 8 stand-in: records which record sets were scored and by which model."""

    def __init__(self, adapted_mrr: float):
        self.adapter = {"trainable_names": ["layers.27.w"]}
        self.adapted_mrr = adapted_mrr
        self.scored: list[tuple[str, tuple[str, ...]]] = []

    def evaluate(self, records, **kw):
        self.scored.append(("adapted", tuple(r["id"] for r in records)))
        return _m(0.7, self.adapted_mrr, len(records))

    def evaluate_base(self, records, **kw):
        self.scored.append(("frozen", tuple(r["id"] for r in records)))
        return _m(0.6, 0.7, len(records))


FRESH = [{"id": f"fresh-{i:04d}", "query": f"fresh {i}", "positive": "d"} for i in range(4)]
REPORT = [{"id": f"test-{i:04d}", "query": f"exposed {i}", "positive": "d"} for i in range(3)]
VAL = [{"id": f"validation-{i:04d}", "query": f"val {i}", "positive": "d"} for i in range(3)]


def section_8_namespace(*, adapted_mrr: float, use_byod: bool, preregistered: bool, keep_frozen: bool = False) -> dict:
    selection = {"settings": dict(samples.GRID_CONFIGS["C0"]), "keep_frozen": keep_frozen, "selected_model": "frozen" if keep_frozen else "adapted C0"}
    return {
        "json": json, "pipe": _Pipe(adapted_mrr), "report_records": REPORT, "val_records": VAL, "val_documents": ["d"],
        "documents": lambda records: ["d"], "document_set": ["d"] * 77, "INSTRUCTION": "i", "floor": _m(0.013, 0.064), "baseline_lexical": _m(0.3, 0.4),
        "frozen_report": _m(0.63, 0.741), "frozen_val": _m(0.6, 0.7), "frozen_vs_floor": "above", "report_label": "exposed development test",
        "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k", "data_source": "stand-in", "dataset_manifests": {}, "disjoint": {},
        "adapt_result": {"history": [], "best_epoch": 0}, "adapt_seconds": 0.0, "near_duplicates": {"near_duplicates": {"test": 9}},
        "USE_BYOD": use_byod, "preregistered": preregistered, "settings_differing": [] if preregistered else ["epochs"], "run_settings": {},
        "grid_selection": selection, "fresh": {"manifest": {"manifest_sha256": "f" * 64}} if not use_byod else None, "fresh_records": FRESH if not use_byod else [],
        "test_look": samples.test_look, "paired_rank_comparison": metrics.paired_rank_comparison,
        "brief": lambda m: {k: m[k] for k in ("recall@1", "mrr")},
    }


def _fresh_scored(namespace: dict) -> bool:
    return any(ids and ids[0].startswith("fresh-") for _model, ids in namespace["pipe"].scored)


# ---- R72-M3 / SCORE_ONCE item 7 ---------------------------------------------------------------------------------


def _settings(**changes) -> dict:
    base = {k: (dict(v) if isinstance(v, dict) else v) for k, v in samples.PREREGISTERED_SETTINGS.items()}
    base["configuration"] = dict(samples.GRID_CONFIGS["C0"])
    base.update(changes)
    return base


SELECTION = {"settings": dict(samples.GRID_CONFIGS["C0"])}


def test_identical_settings_are_preregistered():
    assert samples.preregistration_diff(_settings(), SELECTION) == []


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"epochs": 3}, "epochs"),
        ({"batch_size": 8}, "batch_size"),
        ({"adapt_seed": 1}, "adapt_seed"),
        ({"configuration": {"trainable_layers": 2, "lr": 2e-5}}, "configuration"),
        ({"configuration": {"trainable_layers": 1, "lr": 5e-5}}, "configuration"),
        ({"data_source": "BYOD (pairs.csv)"}, "data_source"),
        ({"split_seed": 7}, "split_seed"),
        ({"instruction": "find the intent"}, "instruction"),
        ({"temperature": 0.2}, "temperature"),
        ({"max_train_tokens": 128}, "max_train_tokens"),
        ({"fresh_test_sha256": "0" * 64}, "fresh_test_sha256"),
        ({"sizes": {"train": 616, "validation": 154, "exposed_development_test": 385, "fresh_test": 0}}, "sizes"),
    ],
)
def test_any_changed_setting_is_exploratory(change, key):
    differs = samples.preregistration_diff(_settings(**change), SELECTION)
    assert key in differs
    assert samples.test_look(not differs, 0, None) == "not scored (exploratory)"


def test_missing_setting_or_no_grid_is_exploratory():
    settings = _settings()
    settings.pop("epochs")
    assert "epochs" in samples.preregistration_diff(settings, SELECTION)
    assert samples.preregistration_diff(_settings(), None) == ["grid (not run in this session)"]


def test_identical_rerun_is_a_reproduction():
    assert samples.test_look(True, 0, None) == "pre-registered evaluation (this runtime)"
    assert samples.test_look(True, 1, None) == "reproduction"
    assert samples.test_look(True, 0, {"run": "hosted T4"}) == "reproduction"
    assert samples.RECORDED_EVALUATION is None  # until the first hosted run is recorded


def test_exploratory_section_8_never_reads_the_fresh_test(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    source = _cell(notebook, "delta_mrr = ")
    namespace = section_8_namespace(adapted_mrr=0.85, use_byod=False, preregistered=False)
    exec(compile(source, "<section 8>", "exec"), namespace)
    assert not _fresh_scored(namespace) and namespace["fresh_result"] is None
    assert namespace["look"] == "not scored (exploratory)" and namespace["run_tag"] == "_exploratory_1"
    report = json.loads((tmp_path / "outputs" / "qwen3_embedding_evaluation_report_exploratory_1.json").read_text(encoding="utf-8"))
    assert report["test_look"] == "not scored (exploratory)" and report["settings_differing"] == ["epochs"]
    assert not (tmp_path / "outputs" / "qwen3_embedding_evaluation_report.json").exists()


def test_preregistered_section_8_scores_the_fresh_test_then_labels_a_rerun_a_reproduction(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    source = _cell(notebook, "delta_mrr = ")
    namespace = section_8_namespace(adapted_mrr=0.85, use_byod=False, preregistered=True)
    exec(compile(source, "<section 8>", "exec"), namespace)
    assert ("frozen", tuple(r["id"] for r in FRESH)) in namespace["pipe"].scored
    assert ("adapted", tuple(r["id"] for r in FRESH)) in namespace["pipe"].scored
    assert namespace["look"] == "pre-registered evaluation (this runtime)" and namespace["run_tag"] == ""
    first = json.loads((tmp_path / "outputs" / "qwen3_embedding_evaluation_report.json").read_text(encoding="utf-8"))
    assert first["fresh_test"]["test_look"] == "pre-registered evaluation (this runtime)" and first["fresh_test"]["paired_mrr"]["n_queries"] == 4
    exec(compile(source, "<section 8 again>", "exec"), namespace)
    assert namespace["look"] == "reproduction" and namespace["run_tag"] == "_reproduction_2"
    again = json.loads((tmp_path / "outputs" / "qwen3_embedding_evaluation_report_reproduction_2.json").read_text(encoding="utf-8"))
    assert again["test_look"] == "reproduction"
    assert json.loads((tmp_path / "outputs" / "qwen3_embedding_evaluation_report.json").read_text(encoding="utf-8")) == first


def test_keep_frozen_scores_only_the_frozen_model_on_the_fresh_test(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    namespace = section_8_namespace(adapted_mrr=0.85, use_byod=False, preregistered=True, keep_frozen=True)
    exec(compile(_cell(notebook, "delta_mrr = "), "<section 8>", "exec"), namespace)
    assert ("adapted", tuple(r["id"] for r in FRESH)) not in namespace["pipe"].scored
    assert namespace["fresh_result"]["adapted"] is None and namespace["fresh_result"]["selected_model"] == "frozen"


def test_no_stop_reads_a_test_split(notebook):
    six = _cell(notebook, "frozen_vs_floor = ")
    assert "frozen_vs_floor = 'above' if frozen_val['mrr'] > val_floor['mrr'] else 'not above'" in six
    eight = _cell(notebook, "delta_mrr = ")
    guard = eight[eight.index("if not USE_BYOD and preregistered"):]
    assert guard.splitlines()[0].endswith("['adapted_vs_frozen_validation_mrr'] != 'improved':")
    nine = _cell(notebook, "pipe.save_artifact(")
    assert "reloaded.evaluate(val_records" in nine and "exposed_records[:77]" not in nine


def test_section_7_computes_preregistered_from_every_setting(notebook):
    source = _cell(notebook, "settings_differing = preregistration_diff(")
    for key in samples.PREREGISTERED_SETTINGS:
        assert f"'{key}':" in source, key
    assert "'configuration': {'trainable_layers': layers, 'lr': lr}" in source
    assert "DEFAULT_SETTINGS" not in _all_text(notebook)


# ---- R72-M1: validation grid -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("val", "frozen", "config", "keep_frozen"),
    [
        ({"C0": 0.80, "C1": 0.81, "C2": 0.815}, 0.70, "C0", False),  # inside the margin: C0 stays
        ({"C0": 0.80, "C1": 0.80, "C2": 0.80}, 0.70, "C0", False),  # exact tie: C0
        ({"C0": 0.80, "C1": 0.825, "C2": 0.83}, 0.70, "C1", False),  # C2 not 0.02 above the best 2-layer
        ({"C0": 0.80, "C1": 0.81, "C2": 0.835}, 0.70, "C2", False),  # C2 beats C0 and C1 by 0.02
        ({"C0": 0.80, "C1": 0.70, "C2": 0.70}, 0.79, "C0", True),  # not 0.02 above frozen: keep frozen
    ],
)
def test_selection_rule(val, frozen, config, keep_frozen):
    selection = samples.select_configuration(val, frozen)
    assert selection["config"] == config and selection["keep_frozen"] is keep_frozen
    assert selection["settings"] == samples.GRID_CONFIGS[config]


def test_grid_cell_runs_the_three_configs_on_validation_only(notebook):
    source = _cell(notebook, "grid_selection = select_configuration(")
    assert "RUN_GRID = True  # @param" in source
    assert "pipe.adapt(train_records, val_records," in source
    for name in ("exposed_records", "report_records", "fresh_records", ".evaluate("):
        assert name not in source, name
    assert samples.GRID_CONFIGS == {"C0": {"trainable_layers": 2, "lr": 5e-5}, "C1": {"trainable_layers": 2, "lr": 2e-5}, "C2": {"trainable_layers": 4, "lr": 2e-5}}


def test_prose_no_longer_claims_independence(notebook):
    text = _all_text(notebook)
    for stale in ("never used for training or epoch selection", "independent test split", "on an independent split", "never tuned for", "not tuned for", "unseen messages"):
        assert stale not in text, stale
    md = _markdown(notebook)
    assert "**exposed development test**" in md and "**Training-data overlap (DAT9).**" in md and "MTEB" in md


# ---- R72-M1 / R72-s2: fresh test --------------------------------------------------------------------------------


def _synthetic_corpus(n_intents: int = 77, train_per: int = 12, test_per: int = 40, seed: int = 3) -> dict:
    rng = random.Random(seed)
    intents = [f"intent_{i:02d}" for i in range(n_intents)]
    train = [{"id": f"train-{k:05d}", "text": f"train message {k} about {it} {rng.random():.6f}", "intent": it} for k, it in enumerate(it for it in intents for _ in range(train_per))]
    test = [{"id": f"test-{k:05d}", "text": f"test message {k} about {it}", "intent": it} for k, it in enumerate(it for it in intents for _ in range(test_per))]
    # one exact repeat (modulo punctuation and case) of a training message in the never-scored pool
    test[-1] = {**test[-1], "text": train[-1]["text"].upper() + "!", "intent": train[-1]["intent"]}
    return {"train": train, "test": test}


def test_fresh_test_excludes_exposed_workshop_and_training_repeats():
    corpus = _synthetic_corpus()
    built = samples.build_fresh_test(corpus, verify=False)
    records, manifest = built["records"], built["manifest"]
    keys = {samples.query_key(r["query"]) for r in records}
    exposed = {samples.query_key(r["query"]) for r in samples.build_sample_dataset(corpus)["test"]}
    workshop = {samples.query_key(r["query"]) for r in samples.workshop_w16_queries(corpus)}
    assert len(records) == 770 and len(workshop) == 154
    assert not keys & exposed and not keys & workshop
    assert manifest["exact_repeats_of_train_csv_dropped"] == 1
    assert samples.query_key(corpus["test"][-1]["text"]) not in keys
    assert all(sum(r["intent"] == it for r in records) == 10 for it in {r["intent"] for r in records})
    assert manifest["manifest_sha256"] == samples.fresh_test_digest(records)
    with pytest.raises(ValueError, match="differs from the pinned"):
        samples.build_fresh_test(corpus)


def test_workshop_draw_replica_matches_the_workshop_cell():
    """The W16 replica in samples.py must pick exactly the queries the workshop's own cell picks."""
    import importlib.util
    from collections import defaultdict

    spec = importlib.util.spec_from_file_location("_w16_source", WORKSHOP_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cells = [c for c in module.CELLS if "rng = random.Random(SAMPLE_SEED)" in c.get("source", "")]
    assert len(cells) == 1
    corpus = _synthetic_corpus()
    namespace = {"random": random, "json": json, "hashlib": hashlib, "SAMPLE_SEED": samples.WORKSHOP_W16_SEED, "MAX_TEXT_CHARS": samples.MAX_TEXT_CHARS, "defaultdict": defaultdict}
    namespace["train_rows"] = [{"source_id": r["id"], "text": r["text"].strip(), "intent": r["intent"]} for r in corpus["train"]]
    namespace["test_rows"] = [{"source_id": r["id"], "text": r["text"].strip(), "intent": r["intent"]} for r in corpus["test"]]
    exec(compile(cells[0]["source"], "<w16 draw>", "exec"), namespace)
    assert sorted(q["query_id"] for q in namespace["queries"]) == sorted(r["id"] for r in samples.workshop_w16_queries(corpus))
    # W16 behaviour is unchanged: its pins are the tutorial's.
    assert samples.WORKSHOP_W16_SEED == 42 and samples.CORPUS_FILES["test"][0] == "test.csv"


def test_fresh_test_pin_is_recorded_in_the_preregistration():
    text = (ROOT / "docs" / "qwen3-embedding-preregistration.md").read_text(encoding="utf-8")
    assert samples.FRESH_TEST_MANIFEST_SHA256 in text and samples.TRANSFER_MANIFEST_SHA256 in text
    assert samples.PREREGISTERED_SETTINGS["fresh_test_sha256"] == samples.FRESH_TEST_MANIFEST_SHA256


# ---- R72-M2: transfer check -------------------------------------------------------------------------------------


def test_transfer_set_draws_two_per_intent_and_is_pinned():
    rows = [[f"utterance {i} for intent {i % 150}", f"intent_{i % 150:03d}"] for i in range(4500)]
    data = json.dumps({"test": rows}).encode("utf-8")
    built = samples.build_transfer_set(data, verify=False)
    assert len(built["records"]) == 300 and len(built["documents"]) == 150
    assert built["manifest_sha256"] == samples.dataset_digest(built["records"])
    with pytest.raises(ValueError, match="differs from the pinned"):
        samples.build_transfer_set(data)
    assert samples.TRANSFER_FILE == ("data_full.json", 2_495_390, "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0")
    assert "828f8093932c8fe6ca7936c3d2e52903b1c523de" in samples.TRANSFER_URL


def test_transfer_cell_is_display_only(notebook):
    source = _cell(notebook, "transfer = build_transfer_set(")
    assert "RUN_TRANSFER_CHECK = True  # @param" in source
    assert "pipe.evaluate_base(transfer['records']" in source and "paired_rank_comparison(" in source
    for word in ("grid_selection", "preregistered =", "raise RuntimeError(f"):
        assert word not in source, word


def test_paired_rank_comparison_is_seeded_and_brackets_the_mean():
    a, b = [1, 2, 3, 4, 1, 2] * 10, [1, 1, 2, 4, 1, 3] * 10
    first = metrics.paired_rank_comparison(a, b)
    assert first == metrics.paired_rank_comparison(a, b)
    expected = sum(1 / y - 1 / x for x, y in zip(a, b, strict=True)) / len(a)
    assert first["delta_mrr"] == pytest.approx(expected) and first["ci"][0] <= expected <= first["ci"][1]
    assert first["n_improved"] + first["n_worse"] + first["n_unchanged"] == len(a)
    with pytest.raises(ValueError):
        metrics.paired_rank_comparison([1], [1, 2])


def test_evaluate_base_without_an_adapter_is_evaluate():
    from qwen3_embedding_pipeline.pipeline import Qwen3EmbeddingPipeline

    calls = []
    pipe = types.SimpleNamespace(adapter=None, _model=None, evaluate=lambda records, **kw: calls.append(kw) or {"mrr": 1.0})
    assert Qwen3EmbeddingPipeline.evaluate_base(pipe, [], instruction="i") == {"mrr": 1.0} and calls == [{"instruction": "i", "candidates": None}]


# ---- R72-m3: spec 2.5, DOC17, DAT20 -----------------------------------------------------------------------------


def test_use_notice_precedes_the_first_code_cell(notebook):
    first_code = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code")
    assert "> **⚠️ Notice.** For research, training and evaluation only." in _source(notebook["cells"][0]) and first_code > 0
    assert "`DIMER_ISOLATED_ENV`" in _markdown(notebook)


def test_byod_refusal_is_a_message_and_drops_the_earlier_split(notebook):
    source = _cell(notebook, "if USE_BYOD:")
    assert source.index("globals().pop(_stale, None)") < source.index("if USE_BYOD:")
    refused = samples.DatasetRefused("BYOD refused: x")
    assert refused.dimer_refusal is True and refused._render_traceback_() == ["BYOD refused: x"]
    generator = (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")
    assert 'if getattr(exc, "dimer_refusal", False):' in generator and "class IsolatedCellRefusal(RuntimeError):" in generator


# ---- R72-m2: provenance stamp -------------------------------------------------------------------------------------


def test_stamp_revision_holds_the_carried_modules(notebook):
    revision = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    try:
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{revision}^{{commit}}"], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"stamp revision {revision[:12]} is not in this clone (shallow checkout)")
    for cell in notebook["cells"]:
        meta = cell.get("metadata", {}).get("dimer", {})
        if cell["cell_type"] != "code" or not meta.get("embedded_module"):
            continue
        rel = meta["embedded_module"]
        committed = subprocess.run(["git", "-C", str(ROOT), "show", f"{revision}:{rel}"], check=True, capture_output=True).stdout.decode("utf-8")
        body = [line for line in committed.splitlines() if line.strip() and not line.startswith("from .")]
        carried = _source(cell)
        assert all(line in carried for line in body), rel
