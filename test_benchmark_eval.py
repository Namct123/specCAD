import json

import pytest

from benchmark.eval_sessions import BenchmarkEvaluator, SessionLogParser


def test_cumulative_session_usage(tmp_path):
    log = tmp_path / "usage.jsonl"
    counters = [(100, 40, 10), (140, 60, 18)]
    events = []
    before = (0, 0, 0)
    for counter in counters:
        fields = ("input_tokens", "cached_input_tokens", "output_tokens")
        events.append({"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": dict(zip(fields, counter)),
            "last_token_usage": dict(zip(fields, (a - b for a, b in zip(counter, before)))),
        }}})
        before = counter
    log.write_text("\n".join(map(json.dumps, events))+"\n", encoding="utf-8")
    metrics = SessionLogParser.parse_log(log)
    assert (metrics.uncached_input_tokens, metrics.output_tokens, metrics.total_tokens) == (80, 18, 158)
    assert metrics.model_calls_with_usage == 2


def test_ollama_usage_is_separate_and_missing_files_are_rejected(tmp_path):
    log = tmp_path / "usage.jsonl"
    log.write_text(json.dumps({"prompt_eval_count": 40, "eval_count": 20}) + "\n", encoding="utf-8")
    metrics = SessionLogParser.parse_log(log)

    assert metrics.total_tokens == 0
    assert (metrics.ollama_input_tokens, metrics.ollama_output_tokens) == (40, 20)
    with pytest.raises(ValueError, match="measured benchmark"):
        BenchmarkEvaluator(direct_log=log).run_evaluation()
