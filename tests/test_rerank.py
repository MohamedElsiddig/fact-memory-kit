"""Tests for the opt-in cross-encoder re-ranker (no network: clients are fakes)."""
import json
import re

import pytest

from factkit.rerank import (
    bench,
    enabled_for,
    parse_scores,
    rerank,
    score_pairs,
)


# --------------------------------------------------------------- fake clients
class ScriptedChat:
    """Returns scripted answers; records every prompt."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def chat(self, prompt):
        self.prompts.append(prompt)
        index = len(self.prompts) - 1
        return self.answers[index] if index < len(self.answers) else "[]"


class CompleteOnly:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.answer


class Boom:
    def __init__(self):
        self.calls = 0

    def chat(self, prompt):
        self.calls += 1
        raise RuntimeError("endpoint down")


class SlowChat:
    def __init__(self, delay=0.05, answer="[0.5, 0.5]"):
        self.delay = delay
        self.answer = answer
        self.calls = 0

    def chat(self, prompt):
        import time
        self.calls += 1
        time.sleep(self.delay)
        return self.answer


class KeywordChat:
    """Scores a document 1.0 when it contains a keyword, else a default."""

    def __init__(self, keywords, default=0.0):
        self.keywords = set(keywords)
        self.default = default
        self.prompts = []

    def chat(self, prompt):
        self.prompts.append(prompt)
        docs = [m.group(2) for m in re.finditer(r"^DOC (\d+): (.*)$", prompt, re.M)]
        scores = []
        for doc in docs:
            hit = any(word in doc for word in self.keywords)
            scores.append(1.0 if hit else self.default)
        return json.dumps(scores)


def hits(*rows):
    return [{"fact_id": fid, "content": text, "score": score} for fid, text, score in rows]


# -------------------------------------------------------------------- parsing
def test_parse_json_array():
    assert parse_scores("[0.1, 0.9, 0]", 3) == [0.1, 0.9, 0.0]


def test_parse_prose_around_array():
    assert parse_scores("Sure, here are the scores: [0.2, 0.8]. Hope that helps!", 2) == [0.2, 0.8]


def test_parse_fenced_array():
    assert parse_scores("```json\n[0.25, 0.75]\n```", 2) == [0.25, 0.75]


def test_parse_clamps_out_of_range():
    assert parse_scores("[-3, 0.5, 4.2]", 3) == [0.0, 0.5, 1.0]


def test_parse_unparseable_is_zero():
    assert parse_scores("I cannot score these documents.", 2) == [0.0, 0.0]
    assert parse_scores("", 2) == [0.0, 0.0]
    assert parse_scores(None, 2) == [0.0, 0.0]


def test_parse_short_array_pads_with_zero():
    assert parse_scores("[0.4]", 3) == [0.4, 0.0, 0.0]


def test_parse_non_finite_and_junk_entries():
    assert parse_scores('[0.5, "0.7", null, NaN, true]', 5) == [0.5, 0.7, 0.0, 0.0, 0.0]


# ----------------------------------------------------------------- score_pairs
def test_score_pairs_uses_chat():
    client = ScriptedChat(["[0.2, 0.9]"])
    assert score_pairs(client, "q", ["a", "b"]) == [0.2, 0.9]
    assert "DOC 1: a" in client.prompts[0]
    assert "DOC 2: b" in client.prompts[0]


def test_score_pairs_supports_complete_only_client():
    client = CompleteOnly("[0.6, 0.7]")
    assert score_pairs(client, "q", ["a", "b"]) == [0.6, 0.7]


def test_score_pairs_client_error_never_raises():
    client = Boom()
    assert score_pairs(client, "q", ["a", "b", "c"]) == [0.0, 0.0, 0.0]
    assert client.calls == 1          # stops after the failure, no retry storm


def test_score_pairs_batch_splits_and_preserves_order():
    client = ScriptedChat(["[0.1, 0.2]", "[0.3, 0.4]", "[0.5]"])
    out = score_pairs(client, "q", ["a", "b", "c", "d", "e"], batch=2)
    assert out == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert len(client.prompts) == 3
    assert [p.count("DOC ") for p in client.prompts] == [2, 2, 1]


def test_score_pairs_bad_batch_does_not_drop_later_batches():
    client = ScriptedChat(["not json at all", "[0.8, 0.9]"])
    assert score_pairs(client, "q", ["a", "b", "c", "d"], batch=2) == [0.0, 0.0, 0.8, 0.9]


# ---------------------------------------------------------------------- rerank
def test_rerank_too_few_does_not_call_client():
    client = Boom()
    original = hits((1, "only fact", 0.9))
    result = rerank(client, "q", original)
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "too_few"
    assert client.calls == 0
    assert isinstance(result["latency_ms"], float) and result["latency_ms"] >= 0


def test_rerank_empty_hits():
    result = rerank(Boom(), "q", [])
    assert result["reason"] == "too_few" and result["hits"] == []


def test_rerank_timeout_returns_original():
    original = hits((1, "a fact", 0.9), (2, "b fact", 0.8))
    client = SlowChat(delay=0.05)
    result = rerank(client, "q", original, timeout_ms=5)
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "timeout"
    assert result["latency_ms"] >= 5.0


def test_rerank_accepts_when_mean_rr_gain_passes():
    # 1 > 3 > 2 > 4 keeps the top-1 but lifts fact 3 from rank 3 to rank 2.
    original = hits((1, "alpha", 1.0), (2, "bravo", 0.9),
                    (3, "charlie", 0.8), (4, "delta", 0.7))
    client = ScriptedChat(["[0.9, 0.0, 1.0, 0.0]"])
    result = rerank(client, "q", original, relevant={1, 3})
    assert result["reranked"] is True
    assert result["top1_changed"] is False
    assert result["reason"] == "mrr_gain"
    assert [h["fact_id"] for h in result["hits"]] == [1, 3, 2, 4]
    assert result["mrr_after"] > result["mrr_before"]
    # accepted hits are copies carrying the new blended score
    assert result["hits"][1]["score"] > result["hits"][2]["score"]
    assert result["hits"][1]["original_score"] == 0.8
    assert original[2]["score"] == 0.8       # input untouched


def test_rerank_top1_change_is_accepted():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8))
    result = rerank(ScriptedChat(["[0.0, 1.0]"]), "q", original)
    assert result["reranked"] is True
    assert result["reason"] == "top1_change"
    assert [h["fact_id"] for h in result["hits"]] == [2, 1]


def test_rerank_no_change_returns_original():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.7), (3, "charlie", 0.5))
    result = rerank(ScriptedChat(["[0.1, 0.05, 0.0]"]), "q", original)
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "no_gain"


def test_rerank_deterministic_tie_break_by_fact_id():
    original = hits((2, "alpha", 0.9), (1, "bravo", 0.9), (3, "charlie", 0.5))
    result = rerank(ScriptedChat(["[0.1, 0.9, 0.5]"]), "q", original, weights=(1.0, 0.0))
    assert result["reranked"] is True
    assert [h["fact_id"] for h in result["hits"]] == [1, 2, 3]


def test_rerank_k_only_scores_head():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8), (3, "charlie", 0.7))
    client = ScriptedChat(["[0.0, 1.0]"])
    result = rerank(client, "q", original, k=2)
    assert len(client.prompts[0].split("DOC ")) - 1 == 2
    assert [h["fact_id"] for h in result["hits"]] == [2, 1, 3]
    assert result["hits"][2] is original[2]      # tail passed through untouched


def test_rerank_weights_are_configurable():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8), (3, "charlie", 0.7))
    result = rerank(ScriptedChat(["[0.0, 0.2, 1.0]"]), "q", original, weights=(0.0, 1.0))
    assert result["reranked"] is True
    assert [h["fact_id"] for h in result["hits"]] == [3, 2, 1]


def test_rerank_client_error_returns_original():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8))
    result = rerank(Boom(), "q", original)
    assert result["hits"] is original
    assert result["reason"] == "error"
    assert result["reranked"] is False


def test_rerank_latency_is_float():
    result = rerank(ScriptedChat(["[0.5, 0.5]"]), "q", hits((1, "a", 0.9), (2, "b", 0.8)))
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0


# ----------------------------------------------------------------------- bench
def test_bench_math_on_fixture():
    cases = [
        {"query": "laptop ram", "expect": [1],
         "candidates": [{"fact_id": 1, "content": "laptop has 16 GB RAM", "score": 0.9}]},
        {"query": "wifi", "expect": [5],
         "candidates": [
             {"fact_id": 7, "content": "printer is upstairs", "score": 0.9},
             {"fact_id": 5, "content": "wifi password is in the vault", "score": 0.8},
             {"fact_id": 3, "content": "unrelated note", "score": 0.4},
         ]},
        {"query": "nothing", "expect": [9],
         "candidates": [
             {"fact_id": 4, "content": "old note a", "score": 0.6},
             {"fact_id": 2, "content": "old note b", "score": 0.3},
         ]},
    ]
    snapshot = json.loads(json.dumps(cases))
    report = bench(KeywordChat(["wifi"]), cases)

    assert report["n"] == 3 and report["k"] == 5
    assert report["hit_at_k_before"] == pytest.approx(2 / 3)
    assert report["hit_at_k_after"] == pytest.approx(2 / 3)
    assert report["mrr_before"] == pytest.approx(0.5)
    assert report["mrr_after"] == pytest.approx(2 / 3)
    assert isinstance(report["mean_latency_ms"], float)
    assert report["mean_latency_ms"] >= 0.0
    assert report["accepted"] == 1
    # read-only: candidates are copied, not reordered in place
    assert cases == snapshot
    assert [c["fact_id"] for c in cases[1]["candidates"]] == [7, 5, 3]


def test_bench_empty_cases():
    report = bench(Boom(), [])
    assert report["n"] == 0
    assert report["hit_at_k_before"] == report["mrr_after"] == 0.0
    assert report["mean_latency_ms"] == 0.0


# ----------------------------------------------------------------- endpoint gate
def test_enabled_for_requires_budget_on_remote_endpoints():
    assert enabled_for("http://127.0.0.1:1234/v1") is True
    assert enabled_for("http://localhost:1234/v1") is True
    assert enabled_for("https://api.example.com/v1") is False
    assert enabled_for("https://api.example.com/v1", budget_ms=200) is True


# ------------------------------------------------- embedding-mode re-ranking
class VecClient:
    """Maps exact text to a vector; records every embedded text."""

    def __init__(self, vectors, delay=0.0):
        self.vectors = dict(vectors)
        self.delay = delay
        self.calls = []

    def embed_one(self, text):
        self.calls.append(text)
        if self.delay:
            import time
            time.sleep(self.delay)
        return self.vectors[text]


class BatchEmbed:
    """Batch-only client: forces the embed([text])[0] fallback."""

    def __init__(self, vectors):
        self.vectors = dict(vectors)
        self.batches = []

    def embed(self, texts):
        texts = list(texts)
        self.batches.append(texts)
        return [self.vectors[text] for text in texts]


class BoomEmbed:
    def __init__(self):
        self.calls = 0

    def embed_one(self, text):
        self.calls += 1
        raise RuntimeError("embed endpoint down")


def test_rerank_embedding_blend_reorders_and_reports_cross_scores():
    original = hits((1, "alpha doc", 0.9), (2, "bravo doc", 0.8))
    client = VecClient({"q": [1.0, 0.0], "alpha doc": [0.0, 1.0],
                        "bravo doc": [1.0, 0.0]})
    result = rerank(None, "q", original, mode="embedding", embed_client=client)
    assert result["reranked"] is True
    assert result["reason"] == "top1_change"
    assert result["mode"] == "embedding"
    assert [h["fact_id"] for h in result["hits"]] == [2, 1]
    # cosine 0 -> 0.5, cosine 1 -> 1.0, in the original head order
    assert result["cross_scores"] == [pytest.approx(0.5), pytest.approx(1.0)]
    # blended: 0.7 * normalised_original + 0.3 * cross
    assert result["hits"][0]["score"] == pytest.approx(0.7 * (0.8 / 0.9) + 0.3 * 1.0)
    assert result["hits"][1]["score"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.5)
    assert result["hits"][0]["original_score"] == 0.8
    assert result["hits"][0]["cross_score"] == pytest.approx(1.0)
    # the caller's dicts are untouched
    assert original[0]["score"] == 0.9
    assert "original_score" not in original[0]


def test_rerank_embedding_timeout_returns_original():
    original = hits((1, "alpha doc", 0.9), (2, "bravo doc", 0.8))
    client = VecClient({"q": [1.0, 0.0], "alpha doc": [0.0, 1.0],
                        "bravo doc": [1.0, 0.0]}, delay=0.05)
    result = rerank(None, "q", original, mode="embedding",
                    embed_client=client, timeout_ms=1)
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "timeout"
    assert result["mode"] == "embedding"
    assert result["latency_ms"] >= 1.0


def test_rerank_embedding_same_order_is_no_gain():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8), (3, "charlie", 0.7))
    client = VecClient({"q": [1.0, 0.0], "alpha": [1.0, 0.0],
                        "bravo": [0.8, 0.6], "charlie": [0.6, 0.8]})
    result = rerank(None, "q", original, mode="embedding", embed_client=client)
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "no_gain"
    assert result["mode"] == "embedding"
    assert result["cross_scores"] == [
        pytest.approx(1.0), pytest.approx(0.9), pytest.approx(0.8)]


def test_rerank_embedding_missing_client_reports_error():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8))
    result = rerank(None, "q", original, mode="embedding")
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "error"
    assert result["error"] == "no_embed_client"
    assert result["mode"] == "embedding"


def test_rerank_embedding_ignores_non_embedding_client():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8))
    client = ScriptedChat(["[0.0, 1.0]"])
    result = rerank(client, "q", original, mode="embedding")
    assert result["reason"] == "error"
    assert result["error"] == "no_embed_client"
    assert client.prompts == []


def test_rerank_embedding_memoises_query_and_duplicate_docs_per_call():
    original = hits((1, "alpha doc", 0.9), (2, "alpha doc", 0.9),
                    (3, "bravo doc", 0.8))
    client = VecClient({"q": [1.0, 0.0], "alpha doc": [1.0, 0.0],
                        "bravo doc": [0.0, 1.0]})
    rerank(None, "q", original, mode="embedding", embed_client=client)
    assert client.calls == ["q", "alpha doc", "bravo doc"]
    # the memo is per call: a second call embeds query and docs again
    rerank(None, "q", original, mode="embedding", embed_client=client)
    assert client.calls == ["q", "alpha doc", "bravo doc",
                            "q", "alpha doc", "bravo doc"]


def test_rerank_embedding_ties_never_reorder():
    # Equal blended scores: the id tie-break would swap 2 before 1, but a pure
    # tie is not evidence of a better ranking, so the original order stands.
    original = hits((2, "alpha", 0.9), (1, "bravo", 0.9))
    client = VecClient({"q": [1.0, 0.0], "alpha": [1.0, 0.0],
                        "bravo": [1.0, 0.0]})
    result = rerank(None, "q", original, mode="embedding", embed_client=client)
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "no_gain"
    assert [h["fact_id"] for h in result["hits"]] == [2, 1]
    assert result["top1_changed"] is True


def test_rerank_embedding_client_error_returns_original():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8))
    client = BoomEmbed()
    result = rerank(None, "q", original, mode="embedding", embed_client=client)
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "error"
    assert result["error"] == "RuntimeError"
    assert result["mode"] == "embedding"
    assert client.calls == 1


def test_rerank_embedding_batch_only_client_is_supported():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8))
    client = BatchEmbed({"q": [1.0, 0.0], "alpha": [0.0, 1.0],
                         "bravo": [1.0, 0.0]})
    result = rerank(None, "q", original, mode="embedding", embed_client=client)
    assert result["reranked"] is True
    assert [h["fact_id"] for h in result["hits"]] == [2, 1]
    assert client.batches == [["q"], ["alpha"], ["bravo"]]


def test_rerank_embedding_dimension_mismatch_is_reported():
    original = hits((1, "alpha", 0.9), (2, "bravo", 0.8))
    client = VecClient({"q": [1.0, 0.0], "alpha": [1.0, 0.0, 0.0],
                        "bravo": [1.0, 0.0]})
    result = rerank(None, "q", original, mode="embedding", embed_client=client)
    assert result["hits"] is original
    assert result["reranked"] is False
    assert result["reason"] == "error"
    assert result["error"] == "bad_embedding"
