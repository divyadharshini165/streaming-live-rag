"""Run: pytest -q   (no API key needed; the LLM path is tested with a fake streaming client)."""
import json
from types import SimpleNamespace as NS

import pytest

from app.config import Config
from app.engine import StreamingRAGEngine


@pytest.fixture(scope="module")
def eng():
    return StreamingRAGEngine(Config(retrieval_mode="sparse", log_dir="/tmp/srag_test_logs"))


def test_multi_intent_streaming(eng):
    sid = eng.new_session()
    r = eng.run_utterance(sid, "I need to plan a customer workshop in Pune for 30 people, "
                               "and I need the cancellation policy and the catering options.")
    assert r["metrics"]["early_retrieval"]
    assert len(r["sub_queries"]) >= 2
    assert r["citations"] and not r["rejected_citations"]


def test_refine_then_suppress(eng):
    sid = eng.new_session()
    v1 = eng.run_utterance(sid, "Summarize the travel reimbursement rule for an employee trip.")
    v2 = eng.run_utterance(sid, "The trip was international and the booking was made after travel.")
    assert v2["turn_type"] == "late_detail" and v2["answer_version"] == v1["answer_version"] + 1
    assert v2["citation_lineage"]["kept"]
    v3 = eng.run_utterance(sid, "Please repeat your last answer in two bullets.")
    assert v3["turn_type"] == "presentation" and not v3["retrieval_events"]
    assert set(v3["citations"]) <= set(v2["citations"])


def test_out_of_corpus_flags_uncertainty(eng):
    sid = eng.new_session()
    r = eng.run_utterance(sid, "Is there parking available at Orchid Hall?")
    assert r["uncertainty"]


def test_conversational_no_retrieval(eng):
    sid = eng.new_session()
    r = eng.run_utterance(sid, "Hello, can you hear me?")
    assert not r["retrieval_events"] and r["turn_type"] == "conversational"


class FakeCompletions:
    """Streams a reply containing one valid and one fabricated citation."""
    def create(self, **kw):
        if kw.get("response_format"):
            return NS(choices=[NS(message=NS(content=json.dumps({"sub_queries": ["x"]})))],
                      usage=NS(prompt_tokens=10, completion_tokens=5))
        text = "The Training Room seats 35 people [Doc_07 §2]. It is free forever [Doc_99 §9].\nUNCERTAIN: parking."
        return iter([NS(choices=[NS(delta=NS(content=text[i:i + 12]))], usage=None) for i in range(0, len(text), 12)]
                    + [NS(choices=[], usage=NS(prompt_tokens=300, completion_tokens=40))])


def test_llm_path_strips_fabricated_citations(eng):
    eng.llm.client = NS(chat=NS(completions=FakeCompletions()))
    try:
        sid = eng.new_session()
        r = eng.run_utterance(sid, "How many people does the Orchid Hall training room seat?")
        assert "Doc_99 §9" not in r["citations"] and "Doc_99 §9" in r["rejected_citations"]
        assert r["uncertainty"] == "parking."
        assert r["metrics"]["tokens_in"] == 300 and r["metrics"]["cost_usd"] > 0
    finally:
        eng.llm.client = None
