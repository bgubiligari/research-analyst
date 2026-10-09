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
    evidence_table,
    ranked_complaints,
    research_findings,
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
        fetch=fetcher(records, total=100, incomplete=True), classify=lambda q, items: labels, narrate=None)
    assert result.counts == {"collected": 8, "classified": 7, "unclassified": 1,
        "relevant": 6, "excluded": 1, "uncertain": 0, "unclear": 1}
    negative = result.sentiment.set_index("sentiment").loc["negative"]
    assert negative.to_dict() == {"count": 2, "denominator": 5, "percent": 40.0}
    assert result.sentiment["count"].sum() == 5
    assert result.complaints.set_index("theme").loc["debugging"].to_dict() == {
        "count": 6, "denominator": 6, "percent": 100.0}
    assert any("incomplete" in warning for warning in result.warnings)
    assert result.batch.rejected == 1
    for item in result.batch.items:
        assert item.url == f"https://github.com/{REPO}/issues/{item.number}"
        assert item.source_id == f"github:{REPO}:{item.number}"


def test_invalid_ids_duplicates_and_enums():
    result = run_research("q", REPO, START, END, fetch=fetcher([record(1), record(2)]),
        classify=lambda q, items: [label(1), label(1), label(999)], narrate=None)
    assert result.counts["unclassified"] == 2
    result = run_research("q", REPO, START, END, fetch=fetcher([record(1)]),
        classify=lambda q, items: [{"source_id": "x", "relevance": "invented",
                                      "sentiment": "neutral", "complaint_themes": []}], narrate=None)
    assert result.counts["classified"] == 0
    assert evidence_table(result).Sentiment.tolist() == ["unclassified"]
    assert evidence_table(result)["Complaint Themes"].tolist() == ["unclassified"]


def test_empty_skips_model():
    def forbidden(*args):
        pytest.fail("Model called on empty input")
    result = run_research("q", REPO, START, END, fetch=fetcher([]), classify=forbidden, narrate=None)
    assert result.sentiment["count"].sum() == 0
    assert result.sentiment["percent"].isna().all()
    assert result.complaints["count"].sum() == 0
    assert result.complaints["percent"].isna().all()


def test_complaint_prevalence_counts_issues_not_labels():
    labels = [label(1, "unclear"), label(2), label(3),
              label(4, relevance="irrelevant"), label(5, relevance="uncertain")]
    labels[0].complaint_themes = ["debugging", "debugging", "documentation"]
    labels[1].complaint_themes = []
    labels[2].complaint_themes = ["documentation"]
    result = run_research("Narrow synthetic topic", REPO, START, END,
        fetch=fetcher([record(i) for i in range(1, 7)]), classify=lambda q, items: labels, narrate=None)
    themes = result.complaints.set_index("theme")
    assert themes.loc["debugging", "count"] == 1
    assert themes.loc["documentation", "count"] == 2
    assert (themes.denominator == 3).all()
    assert themes.loc["documentation", "percent"] == pytest.approx(200 / 3)


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
        run_research("q", REPO, START, END, fetch=fetcher([], status=status), narrate=None)


def test_model_failure_keeps_real_evidence():
    def failure(*args):
        raise ResearchError("Model unavailable")
    result = run_research("q", REPO, START, END, fetch=fetcher([record(1)]), classify=failure, narrate=None)
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
        assert "not automatically negative" in kwargs["input"][0]["content"]
        assert "complaints actually expressed" in kwargs["input"][0]["content"]
        client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
        with pytest.raises(ResearchError):
            classify_issues("q", batch.items)


def test_streamlit_synthetic_result():
    result = run_research("q", REPO, START, END, fetch=fetcher([record(1)]),
                          classify=lambda q, items: [label(1)], narrate=None)
    with patch("research.run_research", return_value=result) as pipeline:
        app = AppTest.from_file("../app.py", default_timeout=15).run()
        app.button[0].click().run()
    assert not app.exception
    assert pipeline.call_count == 1
    assert len(app.dataframe) == 3
    assert app.dataframe[0].value.equals(result.complaints)
    assert app.dataframe[2].value.iloc[0]["Issue"] == result.batch.items[0].url
    assert "does not filter the GitHub search" in app.info[0].value


def quality_fixture():
    """Illustrative semantic cases with mocked labels, not a model benchmark."""
    texts = ["I love the clear setup; it works beautifully.",
             "Repeated crashes are frustrating and the interface is confusing.",
             "Which version introduced the configuration flag?",
             "The integration is useful, but slow responses are frustrating.",
             "Synthetic missing-label case"]
    labels = [label(1, "positive"), label(2, "negative"), label(3, "neutral"), label(4, "mixed")]
    labels[0].complaint_themes = []
    labels[1].complaint_themes = ["reliability", "usability"]
    labels[2].complaint_themes = []
    labels[3].complaint_themes = ["performance", "usability"]
    return run_research("Synthetic question", REPO, START, END,
        fetch=fetcher([record(i, body=text) for i, text in enumerate(texts, 1)], total=20),
        classify=lambda q, items: list(reversed(labels)), narrate=None)


def test_evidence_mapping_filtering_and_missing_labels():
    result = quality_fixture()
    table = evidence_table(result)
    assert list(table.columns) == ["Issue", "Title", "Sentiment", "Complaint Themes",
                                   "Relevance", "Created At"]
    assert table.Issue.tolist() == [i.url for i in result.batch.items]
    assert table.Sentiment.tolist() == ["positive", "negative", "neutral", "mixed", "unclassified"]
    assert table["Complaint Themes"].tolist() == ["none", "reliability, usability", "none",
                                                "performance, usability", "unclassified"]
    assert table.iloc[-1].Relevance == "unclassified"
    assert evidence_table(result, sentiment="negative", theme="usability").Issue.tolist() == [
        result.batch.items[1].url]
    assert evidence_table(result, sentiment="positive", theme="usability").empty


def test_deterministic_findings_and_descending_themes():
    result = quality_fixture()
    ranked = ranked_complaints(result)
    assert ranked.theme.tolist() == ["usability", "performance", "reliability"]
    assert ranked.percent.tolist() == [50.0, 25.0, 25.0]
    assert research_findings(result) == [
        "Retrieved 5 GitHub issues out of 20 reported matches within the selected date range.",
        ("4 issues were classified; 1 remain unclassified. Of the classified issues, "
        "4 were relevant, 0 irrelevant, and 0 uncertain."),
        ("1 of 4 classifiable relevant issues were labeled negative (25.0%). "
        "0 relevant issues had unclear sentiment and were excluded from this sentiment denominator."),
        ("Most frequently identified complaint themes (up to three): usability — 2 of 4 "
        "relevant classified issues (50.0%); performance — 1 of 4 relevant classified "
        "issues (25.0%); reliability — 1 of 4 relevant classified issues (25.0%)."),
    ]
    assert research_findings(result) == research_findings(result)


def test_empty_and_unclassified_findings():
    for records in ([], [record(1)]):
        result = run_research("q", REPO, START, END,
            fetch=fetcher(records), classify=lambda q, items: [], narrate=None)
        findings = research_findings(result)
        assert "Insufficient classifiable relevant evidence for sentiment findings." in findings
        assert "Insufficient relevant classified evidence for complaint findings." in findings
        assert not any("%" in line for line in findings)
        assert len(evidence_table(result)) == len(records)


def test_filters_do_not_rerun_pipeline():
    result = quality_fixture()
    with patch("research.run_research", return_value=result) as pipeline:
        app = AppTest.from_file("../app.py", default_timeout=15).run()
        app.button[0].click().run()
        app.selectbox[0].select("negative").run()
        app.selectbox[1].select("usability").run()
        app.selectbox[3].select(result.batch.items[2].source_id).run()
        assert pipeline.call_count == 1
    assert not app.exception
    assert len(app.dataframe[2].value) == 1
    assert app.dataframe[2].value.iloc[0]["Issue"] == result.batch.items[1].url
    assert [m.value for m in app.metric] == ["20", "5", "4", "4"]
    assert result.batch.items[2].body in [element.value for element in app.text]


@pytest.mark.parametrize("single_issue", [False, True])
def test_complaint_chart_preserves_pandas_values_and_full_percentage_axis(single_issue):
    result = (run_research("q", REPO, START, END, fetch=fetcher([record(1)]),
                          classify=lambda q, items: [label(1)], narrate=None)
              if single_issue else quality_fixture())
    with patch("research.run_research", return_value=result), patch("streamlit.vega_lite_chart") as chart:
        app = AppTest.from_file("../app.py", default_timeout=15).run()
        app.button[0].click().run()
    assert not app.exception
    data, spec = chart.call_args.args
    expected = ranked_complaints(result)
    assert data[expected.columns].equals(expected)
    assert data.label.tolist() == [f"{value:.1f}%" for value in expected.percent]
    assert spec["encoding"]["x"]["scale"]["domain"] == [0, 100]
    assert spec["encoding"]["y"]["sort"] == data.category.tolist()
    assert spec["autosize"]["type"] == "fit"
    assert chart.call_args.kwargs["width"] == "stretch"
    assert spec["height"] <= 280
    assert all(layer["encoding"]["text"]["field"] == "label" for layer in spec["layer"][1:])
    if single_issue:
        assert data.percent.max() == 100


@pytest.mark.parametrize('selection,expected', [
    ('Relevant', [1, 5]), ('All', [1, 2, 3, 4, 5]),
    ('Irrelevant', [2]), ('Uncertain', [3]),
])
def test_relevance_filter_only_changes_evidence(selection, expected):
    labels = [label(1, 'negative'), label(2, 'negative', 'irrelevant'),
              label(3, 'negative', 'uncertain'), label(5, 'neutral')]
    result = run_research('Synthetic question', REPO, START, END,
        fetch=fetcher([record(i) for i in range(1, 6)]), classify=lambda *args: labels, narrate=None)
    metrics = result.sentiment.copy(), result.complaints.copy(), dict(result.counts), list(result.labels)
    table = evidence_table(result, relevance=selection)
    assert table.Issue.tolist() == [f'https://github.com/{REPO}/issues/{n}' for n in expected]
    assert evidence_table(result, relevance='Relevant', sentiment='negative').Issue.tolist() == [
        f'https://github.com/{REPO}/issues/1']
    with patch('research.run_research', return_value=result) as pipeline:
        app = AppTest.from_file('../app.py', default_timeout=15).run()
        app.button[0].click().run()
        relevance = next(box for box in app.selectbox if box.label == 'Filter by relevance')
        assert relevance.value == 'Relevant'
        assert app.dataframe[2].value.Issue.tolist() == [f'https://github.com/{REPO}/issues/{n}' for n in [1, 5]]
        relevance.select(selection).run()
        assert app.dataframe[2].value.Issue.tolist() == table.Issue.tolist()
        assert pipeline.call_count == 1
    assert not app.exception
    assert result.sentiment.equals(metrics[0]) and result.complaints.equals(metrics[1])
    assert result.counts == metrics[2] and result.labels == metrics[3]
