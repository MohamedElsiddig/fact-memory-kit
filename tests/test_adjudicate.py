"""Tests for optional LLM adjudication of prose contradictions."""
from types import SimpleNamespace

from factkit.adjudicate import MAX_PAIRS, adjudicate, parse_verdict


def cfg():
    return SimpleNamespace(llm_base_url="http://127.0.0.1:1234/v1", llm_model="m", llm_api_key="", llm_timeout=5)


def pair(a="The proxy runs on node A", b="The proxy runs on node B"):
    return {"keep_content": a, "retire_content": b, "keep": 2, "retire": 1}


def test_parse_plain_json():
    v = parse_verdict('{"contradiction": true, "attribute": "host", "evidence": ["node A", "node B"]}')
    assert v["contradiction"] is True and v["attribute"] == "host"


def test_parse_fenced_json():
    v = parse_verdict('```json\n{"contradiction": false, "attribute": "x", "evidence": []}\n```')
    assert v["contradiction"] is False


def test_parse_rejects_non_bool():
    assert parse_verdict('{"contradiction": "yes"}') is None


def test_parse_rejects_garbage():
    assert parse_verdict("I cannot help with that") is None
    assert parse_verdict("") is None


def test_parse_coerces_bad_evidence():
    v = parse_verdict('{"contradiction": true, "evidence": "node A"}')
    assert v["evidence"] == []


def test_adjudicate_keeps_only_contradictions():
    answers = iter(['{"contradiction": true, "attribute": "host", "evidence": ["A", "B"]}',
                    '{"contradiction": false, "attribute": "host", "evidence": []}'])
    out = adjudicate([pair(), pair()], cfg(), post=lambda prompt: next(answers))
    assert len(out) == 1
    assert out[0]["source"] == "llm"
    assert out[0]["verdict"]["attribute"] == "host"


def test_adjudicate_drops_malformed():
    out = adjudicate([pair()], cfg(), post=lambda prompt: "not json")
    assert out == []


def test_adjudicate_stops_on_endpoint_error():
    def boom(prompt):
        raise OSError("connection refused")

    out = adjudicate([pair(), pair()], cfg(), post=boom)
    assert out == []


def test_adjudicate_caps_pairs():
    calls = []

    def post(prompt):
        calls.append(prompt)
        return '{"contradiction": false}'

    adjudicate([pair() for _ in range(MAX_PAIRS + 10)], cfg(), post=post)
    assert len(calls) == MAX_PAIRS


def test_adjudicate_skips_empty_sides():
    out = adjudicate([{"keep_content": "", "retire_content": "x"}], cfg(),
                     post=lambda p: '{"contradiction": true}')
    assert out == []
