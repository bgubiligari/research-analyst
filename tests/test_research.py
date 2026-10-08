"""Synthetic fixtures only; no copied community content."""
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from research import (
    ClassificationBatch,
    Label,
    ResearchError,
    classify_issues,
    run_research,
    search_issues,
)

START, END = date(2026, 9, 1), date(2026, 9, 30)
REPO = "synthetic/repo"


def record(number, **extra):
    return {"number": number, "html_url": f"https://github.com/{REPO}/issues/{number}",
            "created_at": "2026-09-15T12:00:00Z", "title": "Synthetic fixture",
            "body": "Synthetic evidence only", **extra}


def fetcher(records, *, total=None, incomplete=False, status=200):
    def handler(request):
        assert request.url.host == "api.github.com"
        assert request.url.params["q"] == (
            f"repo:{REPO} is:issue is:public created:{START}..{END}")
        assert request.url.params["per_page"] == "10"
        return httpx.Response(status, json={"items": records,
            "total_count": total if total is not None else len(records),
            "incomplete_results": incomplete})

    def fetch(repo, start, end):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            return search_issues(repo, start, end, client=client)
    return fetch


def label(number, sentiment="neutral", relevance="relevant"):
    return Label(source_id=f"github:{REPO}:{number}", relevance=relevance,
                 sentiment=sentiment, complaint_themes=["debugging", "documentation"])


def test_synthetic_end_to_end_same_pipeline():
    records = [record(i) for i in range(1, 9)] + [record(9, pull_request={})]
    labels = [label(1, "negative"), label(2, "negative"), label(3, "positive"),
              label(4), label(5, "mixed"), label(6, "unclear"),
              label(7, relevance="irrelevant")]
    result = run_research("Synthetic question", REPO, START, END,
        fetch=fetcher(records, total=100, incomplete=True), classify=lambda q, items: labels)
    assert result.counts == {"collected": 8, "classified": 7, "unclassified": 1,
        "relevant": 6, "excluded": 1, "uncertain": 0, "unclear": 1}
    negative = result.sentiment.set_index("sentiment").loc["negative"]
    assert negative.to_dict() == {"count": 2, "denominator": 5, "percent": 40.0}
    assert result.sentiment["count"].sum() == 5
    assert any("incomplete" in warning for warning in result.warnings)
    assert result.batch.rejected == 1
    for item in result.batch.items:
        assert item.url == f"https://github.com/{REPO}/issues/{item.number}"
        assert item.source_id == f"github:{REPO}:{item.number}"


def test_invalid_ids_duplicates_and_enums():
    result = run_research("q", REPO, START, END, fetch=fetcher([record(1), record(2)]),
        classify=lambda q, items: [label(1), label(1), label(999)])
    assert result.counts["unclassified"] == 2
    result = run_research("q", REPO, START, END, fetch=fetcher([record(1)]),
        classify=lambda q, items: [{"source_id": "x", "relevance": "invented",
                                      "sentiment": "neutral", "complaint_themes": []}])
    assert result.counts["classified"] == 0


def test_empty_skips_model():
    def forbidden(*args):
        pytest.fail("Model called on empty input")
    result = run_research("q", REPO, START, END, fetch=fetcher([]), classify=forbidden)
    assert result.sentiment["count"].sum() == 0
    assert result.sentiment["percent"].isna().all()


def test_citation_date_dedup_and_cap():
    records = [record(1), record(1), record(2, html_url="https://evil.test"),
               record(3, created_at="2025-01-01T00:00:00Z")]
    records += [record(i) for i in range(4, 20)]
    batch = fetcher(records)(REPO, START, END)
    assert len(batch.items) == 7
    assert batch.rejected == 3


@pytest.mark.parametrize("status", [403, 429, 500, 422])
def test_source_errors(status):
    with pytest.raises(ResearchError):
        run_research("q", REPO, START, END, fetch=fetcher([], status=status))


def test_model_failure_keeps_real_evidence():
    def failure(*args):
        raise ResearchError("Model unavailable")
    result = run_research("q", REPO, START, END, fetch=fetcher([record(1)]), classify=failure)
    assert result.counts["unclassified"] == 1
    assert "Model unavailable" in result.warnings


def test_classifier_sdk_boundary(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "synthetic-test-model")
    batch = fetcher([record(1)])(REPO, START, END)
    with patch("research.OpenAI") as factory:
        client = factory.return_value.__enter__.return_value
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed=ClassificationBatch(items=[label(1)]))
        assert classify_issues("q", batch.items) == [label(1)]
        kwargs = client.responses.parse.call_args.kwargs
        assert kwargs["store"] is False
        assert kwargs["text_format"] is ClassificationBatch
        assert "untrusted" in kwargs["input"][0]["content"]
        client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
        with pytest.raises(ResearchError):
            classify_issues("q", batch.items)


def test_streamlit_synthetic_result():
    result = run_research("q", REPO, START, END, fetch=fetcher([record(1)]),
                          classify=lambda q, items: [label(1)])
    with patch("research.run_research", return_value=result) as pipeline:
        app = AppTest.from_file("../app.py", default_timeout=15).run()
        app.button[0].click().run()
    assert not app.exception
    assert pipeline.call_count == 1
    assert len(app.dataframe) == 2
    assert app.dataframe[1].value.iloc[0]["url"] == result.batch.items[0].url
