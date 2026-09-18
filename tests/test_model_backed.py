"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): retrieval
metrics of the frozen embedder against the lexical baseline, a one-epoch contrastive adaptation of the last
decoder layer on a dozen pairs, and the artifact round trip. Skipped when the weights are absent."""

from __future__ import annotations

import json

import pytest

from qwen3_embedding_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, Qwen3EmbeddingPipeline

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

PAIRS = [
    ("I still have not received my new card, when will it arrive?", "card arrival"),
    ("My card never showed up in the post.", "card arrival"),
    ("How do I change the PIN on my card?", "change pin"),
    ("I forgot my PIN and want a new one.", "change pin"),
    ("What exchange rate do you use for euros?", "exchange rate"),
    ("Is the rate for converting to dollars competitive?", "exchange rate"),
    ("The ATM ate my card and I cannot get it back.", "atm support"),
    ("Which cash machines can I use abroad?", "atm support"),
    ("My payment at the shop was declined.", "declined card payment"),
    ("Why was my card refused at the checkout?", "declined card payment"),
    ("I want to close my account for good.", "terminate account"),
    ("How do I delete my account?", "terminate account"),
]
RECORDS = [{"id": f"p{i:02d}", "query": q, "positive": d} for i, (q, d) in enumerate(PAIRS)]
INSTRUCTION = "Given a customer support message, retrieve the banking intent it expresses"


@pytest.fixture(scope="module")
def pipe():
    return Qwen3EmbeddingPipeline.from_pretrained(device="cpu")


def test_frozen_retrieval_beats_the_lexical_baseline(pipe):
    metrics = pipe.evaluate(RECORDS, instruction=INSTRUCTION)
    baseline = pipe.lexical_baseline(RECORDS)
    assert metrics["n_queries"] == 12 and metrics["n_documents"] == 6 and metrics["adapted"] is False
    assert metrics["mrr"] >= baseline["mrr"] and metrics["recall@1"] >= 0.5


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(
        RECORDS[:8], RECORDS[8:], instruction=INSTRUCTION, epochs=1, trainable_layers=1, batch_size=4
    )
    assert result["n_trainable"] == 15_730_944 and result["history"][0]["note"] == "frozen model"
    assert all(name.startswith("layers.27.") for name in result["trainable_names"])
    assert result["n_total"] == 595_776_512 and "mrr" in result["history"][1]["val"]
    assert result["history"][1]["train_loss"] > 0.0 and result["instruction"] == INSTRUCTION
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert (
        len(manifest["tensors"]) == len(result["trainable_names"])
        and manifest["adapter"]["instruction"] == INSTRUCTION
    )
    reloaded = Qwen3EmbeddingPipeline.from_artifact(artifact, device="cpu")
    texts = [r["query"] for r in RECORDS[:3]]
    assert (
        reloaded.embed(texts, kind="query", instruction=INSTRUCTION)["embeddings"]
        == pipe.embed(texts, kind="query", instruction=INSTRUCTION)["embeddings"]
    )
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
