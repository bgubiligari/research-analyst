"""Synthetic policy contracts; mocked responses do not measure model accuracy."""
import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from research import (
    CLASSIFICATION_INSTRUCTIONS,
    ClassificationBatch,
    Issue,
    Label,
    SearchBatch,
    classify_issues,
    run_research,
)

# These are permitted example labels, not deterministic live-model expectations.
CASES = [
    ("ordinary_bug", "Calling this function raises an unexpected exception.",
     "negative", ["reliability"], "A bug report or stack trace alone is not a debugging complaint."),
    ("debugging", "The workflow fails, and there is no useful trace or error information to identify the cause.",
     "negative", ["debugging", "reliability"], "difficulty diagnosing or troubleshooting failures"),
    ("usability", "The API requires several confusing configuration steps for a simple operation.",
     "negative", ["usability"], "confusing APIs/configuration, cumbersome workflows"),
    ("multiple", "Production requests fail intermittently, and diagnosing failures is difficult because execution traces are missing.",
     "negative", ["reliability", "debugging"], "Multiple labels require evidence for"),
    ("feature", "Please add an optional export method to support this format.",
     "neutral", [], "A feature request is not automatically negative"),
    ("integration", "Messages are corrupted when converted for a third-party model provider.",
     "negative", ["integrations", "reliability"], "A provider name alone is not evidence"),
    ("question", "How do I configure the streaming callback?",
     "neutral", [], "A technical question is not automatically a complaint."),
    ("mixed", "The framework is excellent for prototyping, but its workflow API is very difficult to use in production.",
     "mixed", ["usability"], "meaningful positive and negative perspectives"),
]


def synthetic_issue(number, body):
    return Issue(source_id=f"github:synthetic/policy:{number}", number=number,
                 url=f"https://github.com/synthetic/policy/issues/{number}",
                 title="Synthetic policy example", body=body, created_at="2026-09-01T00:00:00Z")


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_policy_example_schema_and_sdk_request(case, monkeypatch):
    _, body, sentiment, themes, required_policy = case
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "synthetic-test-model")
    issue = synthetic_issue(1, body)
    expected = Label(source_id=issue.source_id, relevance="relevant",
                     sentiment=sentiment, complaint_themes=themes)
    with patch("research.OpenAI") as factory:
        client = factory.return_value.__enter__.return_value
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed=ClassificationBatch(items=[expected]))
        actual = classify_issues("What complaints appear?", [issue])
        request = client.responses.parse.call_args.kwargs
        assert client.responses.parse.call_count == 1
        assert request["input"][0]["content"] == CLASSIFICATION_INSTRUCTIONS
        assert required_policy in request["input"][0]["content"]
        assert json.loads(request["input"][1]["content"])["issues"][0]["body"] == body
        assert request["store"] is False
    assert actual[0].complaint_themes == themes
    assert isinstance(actual[0].complaint_themes, list)
    assert actual[0].sentiment == sentiment


def test_policy_examples_aggregate_without_changing_denominators():
    issues, labels = [], []
    for number, (_, body, sentiment, themes, _) in enumerate(CASES, 1):
        issue = synthetic_issue(number, body)
        issues.append(issue)
        labels.append(Label(source_id=issue.source_id, relevance="relevant",
                            sentiment=sentiment, complaint_themes=themes))
    result = run_research("What complaints appear?", "synthetic/policy",
        date(2026, 9, 1), date(2026, 9, 2),
        fetch=lambda *args: SearchBatch(issues, 8, False), classify=lambda *args: labels)
    sentiment = result.sentiment.set_index("sentiment")
    assert sentiment.loc["negative", "count"] == 5
    assert sentiment.loc["negative", "percent"] == 62.5
    themes = result.complaints.set_index("theme")
    assert (themes.denominator == 8).all()
    assert themes.loc["reliability", "count"] == 4
    assert themes.loc["debugging", "count"] == 2
    assert themes.loc["usability", "percent"] == 25.0
    assert themes.loc["integrations", "percent"] == 12.5


@pytest.mark.parametrize("themes", ["reliability", ["invented"], None])
def test_theme_schema_rejects_single_categories_and_invalid_labels(themes):
    with pytest.raises(ValidationError):
        Label(source_id="synthetic", relevance="relevant", sentiment="neutral", complaint_themes=themes)


def test_policy_keeps_uncertainty_and_independent_sentiment():
    for required in ["Evaluate sentiment and complaint themes separately",
                     "preserve uncertainty", "adversely affects the user or expected functionality",
                     "not template checklists", "do not infer debugging or usability",
                     "never use it as a default", "Positive and neutral issues can have zero themes",
                     "Do not target any sentiment distribution"]:
        assert required in CLASSIFICATION_INSTRUCTIONS


def test_positive_requires_praise_of_existing_framework():
    assert "Successful reproduction/control experiments" in CLASSIFICATION_INSTRUCTIONS
    assert "willingness to contribute are not praise of the existing framework" in CLASSIFICATION_INSTRUCTIONS
