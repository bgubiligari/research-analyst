"""Synthetic historical fixtures; no live source content."""
from datetime import date
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from research import (
    Label,
    ResearchError,
    collect_historical,
    completed_month_range,
    historical_chart_data,
    month_windows,
    run_research,
    sampling_intervals,
)

START, END = date(2026, 4, 1), date(2026, 9, 30)
REPO = "synthetic/repo"


class FakeTime:
    def __init__(self):
        self.now = 0
        self.waits = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def fixture_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def response(items=None, total=100, headers=None, status=200, incomplete=False):
    return httpx.Response(status, headers=headers, json={"items": items or [],
        "total_count": total, "incomplete_results": incomplete})


def raw(number, created):
    return {"number": number, "html_url": f"https://github.com/{REPO}/issues/{number}",
            "created_at": created, "title": "Synthetic fixture", "body": "Synthetic evidence"}


def test_completed_months_and_boundaries():
    assert completed_month_range(date(2026, 10, 8)) == (START, END)
    assert completed_month_range(date(2024, 3, 1)) == (date(2023, 9, 1), date(2024, 2, 29))
    windows = month_windows(date(2024, 2, 10), date(2024, 3, 2))
    assert windows == [(date(2024, 2, 10), date(2024, 2, 29), True),
                       (date(2024, 3, 1), date(2024, 3, 2), True)]
    with pytest.raises(ResearchError):
        month_windows(date(2025, 1, 1), END)


def test_quotas_and_disjoint_intervals():
    intervals = sampling_intervals(START, END)
    assert len(intervals) == 12
    assert [sum(i["quota"] for i in intervals if i["month"] == f"2026-{m:02}")
            for m in range(4, 10)] == [9, 9, 8, 8, 8, 8]
    for early, late in zip(intervals[::2], intervals[1::2], strict=True):
        assert (late["start"] - early["end"]).days == 1
        assert early["order"] == "asc" and late["order"] == "desc"
    assert len(sampling_intervals(START, START)) == 1


def test_synthetic_historical_production_pipeline():
    calls, model_calls = [], []
    def handler(request):
        calls.append(request)
        query = request.url.params["q"]
        first = query.split("created:")[1].split("..")[0]
        month = int(first[5:7])
        limit = int(request.url.params["per_page"])
        # April gets five eligible labels overall; May is sparse; June fails.
        if month == 6:
            return response(status=500)
        count = min(limit, 3 if month == 4 else 1 if month == 5 else limit)
        offset = len(calls) * 100
        records = [raw(offset+i, first+"T12:00:00Z") for i in range(count)]
        return response(records, incomplete=month == 8)
    def classify(question, items):
        model_calls.append(items)
        labels = []
        for i in items:
            if i.created_at.month == 4 and i.number == 202:
                labels.append(Label(source_id=i.source_id, relevance="relevant",
                                    sentiment="unclear", complaint_themes=[]))
            else:
                labels.append(Label(source_id=i.source_id, relevance="relevant",
                    sentiment="negative" if i.number in (100, 200) else "neutral", complaint_themes=[]))
        return labels
    with fixture_client(handler) as client:
        result = run_research("Synthetic question", REPO, START, END, historical=True,
            historical_fetch=lambda r, s, e, **kw: collect_historical(r, s, e, client=client, **kw),
            classify=classify)
    assert len(calls) == 12
    assert len(model_calls) <= 5 and all(len(g) <= 10 for g in model_calls)
    assert len(result.batch.items) <= 50
    april = result.monthly.iloc[0]
    assert april.denominator == 5 and april.negative == 2 and april.negative_percent == 40
    assert april.unclear == 1
    june = result.monthly.iloc[2]
    assert june.errors == 2 and pd.isna(june.reported_matches)
    assert june.denominator == 0 and pd.isna(june.negative_percent)
    assert result.batch.total is None
    chart = historical_chart_data(result.monthly)
    assert "2026-05" not in chart.month.tolist()
    assert chart[chart.month == "2026-04"].series.iloc[0] != chart[chart.month == "2026-07"].series.iloc[0]
    with patch("research.run_research", return_value=result) as pipeline, patch("streamlit.vega_lite_chart") as render:
        app = AppTest.from_file("../app.py", default_timeout=15).run()
        assert app.date_input[0].value == completed_month_range()[0]
        app.button[0].click().run()
        assert not app.exception
        assert app.dataframe[0].value.equals(result.monthly)
        args = render.call_args_list[0].args
        assert args[0].equals(chart)
        assert args[1]["encoding"]["detail"]["field"] == "series"
        app.selectbox[0].select("negative").run()
        assert pipeline.call_count == 1


def test_full_budget_and_failed_batch():
    number = 0
    def handler(request):
        nonlocal number
        first = request.url.params["q"].split("created:")[1].split("..")[0]
        items = []
        for _ in range(int(request.url.params["per_page"])):
            number += 1
            items.append(raw(number, first+"T00:00:00Z"))
        return response(items)
    groups = []
    def classify(q, items):
        groups.append(items)
        raise ResearchError("Synthetic model failure")
    with fixture_client(handler) as client:
        result = run_research("q", REPO, START, END, historical=True, classify=classify,
            historical_fetch=lambda r, s, e, **kw: collect_historical(r, s, e, client=client, **kw))
    assert len(result.batch.items) == 50
    assert len(groups) == 5 and all(len(g) == 10 for g in groups)
    assert result.counts["unclassified"] == 50
    assert historical_chart_data(result.monthly).empty


@pytest.mark.parametrize("budget, expected", [(100, 12), (10, 1)])
def test_rate_headers_wait_or_stop(budget, expected):
    clock, calls = FakeTime(), []
    def handler(request):
        calls.append(request)
        return response(headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "20"}
                        if len(calls) == 1 else {})
    with fixture_client(handler) as client:
        batch, intervals = collect_historical(REPO, START, END, client=client, seconds=budget,
            monotonic=clock.clock, wall=clock.clock, sleep=clock.sleep)
    assert len(calls) == expected
    if expected == 1:
        assert not clock.waits and sum(i["status"] == "unattempted" for i in intervals) == 11
        assert batch.total is None
    else:
        assert sum(clock.waits) == 21


def test_rate_failure_preserves_partial_and_no_retry():
    calls = []
    def handler(request):
        calls.append(request)
        return response([raw(1, "2026-04-01T00:00:00Z")]) if len(calls) == 1 else response(status=429, headers={"retry-after": "60"})
    with fixture_client(handler) as client:
        batch, intervals = collect_historical(REPO, START, END, client=client)
    assert len(calls) == 2 and len(batch.items) == 1
    assert intervals[1]["status"] == "error"
    assert sum(i["status"] == "unattempted" for i in intervals) == 10


def test_identity_validation_utc_and_unknown_labels():
    records = [raw(1, "2026-05-01T00:30:00+01:00"), raw(1, "2026-04-30T23:30:00Z"),
               raw(2, "2026-05-01T00:00:00Z"),
               {**raw(3, "2026-04-30T00:00:00Z"), "pull_request": {}},
               {**raw(4, "2026-04-30T00:00:00Z"), "html_url": "https://evil.test"}]
    with fixture_client(lambda req: response(records)) as client:
        batch, intervals = collect_historical(REPO, date(2026, 4, 30), date(2026, 4, 30), client=client)
    assert len(batch.items) == 1 and batch.rejected == 4
    assert intervals[0]["partial"]


def test_retry_after_pacing_and_remaining_headers():
    clock, calls = FakeTime(), []
    def handler(request):
        calls.append(request)
        return response(headers={"x-ratelimit-remaining": "2", "x-ratelimit-reset": "12",
                                 "retry-after": "8"} if len(calls) == 1 else {})
    with fixture_client(handler) as client:
        collect_historical(REPO, START, END, client=client, monotonic=clock.clock,
                           wall=clock.clock, sleep=clock.sleep)
    assert len(calls) == 12 and sum(clock.waits) == 8


def test_monthly_exclusions_and_batch_identity():
    def handler(request):
        first = request.url.params["q"].split("created:")[1].split("..")[0]
        return response([raw(i, first+"T00:00:00Z") for i in range(1, 6)])
    def classifier(question, items):
        return [Label(source_id=items[0].source_id, relevance="irrelevant", sentiment="negative", complaint_themes=[]),
                Label(source_id=items[1].source_id, relevance="uncertain", sentiment="negative", complaint_themes=[]),
                Label(source_id=items[2].source_id, relevance="relevant", sentiment="unclear", complaint_themes=[]),
                Label(source_id="github:unknown:999", relevance="relevant", sentiment="positive", complaint_themes=[])]
    with fixture_client(handler) as client:
        result = run_research("q", REPO, START, START, historical=True, classify=classifier,
            historical_fetch=lambda r, s, e, **kw: collect_historical(r, s, e, client=client, **kw))
    row = result.monthly.iloc[0]
    assert (row.irrelevant, row.uncertain, row.unclear, row.unclassified) == (1, 1, 1, 2)
    assert row.denominator == 0 and pd.isna(row.negative_percent)
