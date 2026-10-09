"""Synthetic fixtures exercise TB3-A through the production orchestration path."""
from datetime import date
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from research import (
    GapDecision,
    Label,
    ResearchError,
    evaluate_gap,
    gap_context,
    historical_chart_data,
    run_research,
)

REPO = 'synthetic/repo'
START, END = date(2026, 4, 1), date(2026, 9, 30)


def decision(**changes):
    return dict(investigate=True, theme='reliability', reason='insufficient_examples',
                source_ids=[], **changes)


def fixture(*, evaluate=None, targeted=None, failure=False, clock=None, classify_failure=False):
    requests, batches = [], []

    def handler(request):
        requests.append(request)
        query = request.url.params['q']
        first = query.split('created:')[1].split('..')[0]
        if 'in:title,body' in query:
            assert query == f'repo:{REPO} is:issue is:public created:{START}..{END} failure in:title,body'
            assert request.url.params['per_page'] == '10'
            if failure:
                return httpx.Response(500)
            records = targeted if targeted is not None else [raw(999, first)]
        else:
            records = [raw(len(requests)*100+i, first)
                       for i in range(int(request.url.params['per_page']))]
        return httpx.Response(200, json={'items': records, 'total_count': 100,
                                        'incomplete_results': False})

    def classify(question, items):
        batches.append([i.source_id for i in items])
        if classify_failure and len(batches) > 5:
            raise ResearchError('Synthetic failure')
        return [Label(source_id=i.source_id, relevance='relevant', sentiment='negative',
                      complaint_themes=['reliability']) for i in items]

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = run_research('Explain reliability complaints', REPO, START, END,
            historical=True, investigate=evaluate is not None, evaluate=evaluate,
            classify=classify, client=client,
            **({'monotonic': clock} if clock else {}))
    return result, requests, batches


def raw(number, first='2026-04-01', **extra):
    return {'number': number, 'html_url': f'https://github.com/{REPO}/issues/{number}',
            'created_at': first+'T12:00:00Z', 'title': 'Synthetic fixture',
            'body': 'Synthetic failure example', **extra}


def test_sufficient_evidence_skips_search():
    result, requests, batches = fixture(evaluate=lambda *args: {
        'investigate': False, 'theme': None, 'reason': 'sufficient_evidence', 'source_ids': []})
    assert len(requests) == 12 and len(batches) == 5
    assert result.investigation.status == 'skipped'
    assert result.investigation.model_calls == 6
    assert result.investigation.github_requests == 0


def test_one_targeted_search_and_baseline_identity():
    baseline, _, _ = fixture()
    result, requests, batches = fixture(evaluate=lambda *args: decision())
    assert len(requests) == 13 and len(batches) == 6
    assert result.investigation.model_calls == 7
    assert result.investigation.github_requests == 1
    assert result.investigation.status == 'complete'
    assert len(result.batch.items) == 50
    assert baseline.counts == result.counts
    assert baseline.labels == result.labels
    for name in ('sentiment', 'complaints', 'monthly'):
        pd.testing.assert_frame_equal(getattr(baseline, name), getattr(result, name))
    pd.testing.assert_frame_equal(historical_chart_data(baseline.monthly),
                                 historical_chart_data(result.monthly))
    assert result.investigation.supporting[0].url == f'https://github.com/{REPO}/issues/999'
    assert result.investigation.supporting[0] in result.investigation.batch.items
    assert all(len(batch) <= 10 for batch in batches)


@pytest.mark.parametrize('invalid', [
    {'investigate': True, 'theme': 'invented', 'reason': 'insufficient_examples', 'source_ids': []},
    {'investigate': 'true', 'theme': 'reliability', 'reason': 'insufficient_examples', 'source_ids': []},
    {'investigate': True, 'theme': 'reliability', 'reason': 'insufficient_examples', 'source_ids': ['fake']},
    {'investigate': True, 'theme': 'performance', 'reason': 'insufficient_examples', 'source_ids': []},
    {'investigate': True, 'theme': 'reliability', 'reason': 'insufficient_examples', 'source_ids': [], 'query': 'evil'},
    {'investigate': False, 'theme': 'reliability', 'reason': 'sufficient_evidence', 'source_ids': []},
])
def test_invalid_decision_cannot_search(invalid):
    result, requests, _ = fixture(evaluate=lambda *args: invalid)
    assert len(requests) == 12
    assert result.investigation.status == 'skipped'
    assert 'invalid decision' in result.investigation.reason


def test_duplicates_reuse_baseline_and_cap_candidates():
    result, requests, batches = fixture(evaluate=lambda *args: decision(),
        targeted=[raw(100)] + [raw(i) for i in range(9000, 9021)])
    assert len(requests) == 13
    assert len(result.investigation.batch.items) == 10
    assert result.investigation.duplicates == 1
    assert len(batches[-1]) == 9 and 'github:synthetic/repo:100' not in batches[-1]
    assert result.labels[0].source_id == 'github:synthetic/repo:100'
    result, _, batches = fixture(evaluate=lambda *args: decision(), targeted=[raw(100)])
    assert len(batches) == 5 and result.investigation.model_calls == 6
    assert result.investigation.status == 'no_new_support'


@pytest.mark.parametrize('failure', ['retrieval', 'classification', 'empty', 'invalid_sources'])
def test_failures_preserve_baseline(failure):
    baseline, _, _ = fixture()
    targeted = [] if failure == 'empty' else [raw(999, html_url='https://evil.test'),
        raw(1000, first='2025-01-01'), raw(1001, pull_request={})] if failure == 'invalid_sources' else None
    result, _, _ = fixture(evaluate=lambda *args: decision(), targeted=targeted,
        failure=failure == 'retrieval', classify_failure=failure == 'classification')
    assert not result.investigation.supporting
    assert result.investigation.limitations
    pd.testing.assert_frame_equal(result.monthly, baseline.monthly)


def test_expired_budget_skips_evaluation():
    now = [0]
    def evaluate(*args):
        pytest.fail('No evaluation after deadline')
    def clock():
        return now[0]
    # Advancing the clock after baseline classification exhausts both time budgets.
    with patch('research.gap_context', side_effect=lambda q, r: (
            now.__setitem__(0, 451) or ['reliability'], [{'source_id': r.labels[0].source_id}])):
        result, requests, batches = fixture(evaluate=evaluate, clock=clock)
    assert len(requests) == 12 and len(batches) == 5
    assert result.investigation.model_calls == 5
    assert 'budget' in result.investigation.reason


def test_expired_budget_after_evaluation_prevents_search():
    now = [0]
    def evaluate(*args):
        now[0] = 451
        return decision()
    result, requests, _ = fixture(evaluate=evaluate, clock=lambda: now[0])
    assert len(requests) == 12 and result.investigation.model_calls == 6
    assert 'budget' in result.investigation.reason


def test_bounded_evaluation_context_and_sdk(monkeypatch):
    result, _, _ = fixture()
    for issue in result.batch.items:
        issue.title = 'x' * 1000
        issue.body = 'y' * 10000
    themes, examples = gap_context('reliability', result)
    assert len(examples) <= 6
    assert all(len(e['title']) == 200 and len(e['body']) == 1000 for e in examples)
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-test-key')
    monkeypatch.setenv('OPENAI_MODEL', 'synthetic-test-model')
    with patch('research.OpenAI') as factory:
        client = factory.return_value.__enter__.return_value
        client.responses.parse.return_value.output_parsed = GapDecision(**decision())
        evaluate_gap('q', examples, result.counts, themes)
        kwargs = client.responses.parse.call_args.kwargs
        assert kwargs['store'] is False and kwargs['text_format'] is GapDecision
        assert factory.call_args.kwargs['max_retries'] == 0
        assert 'untrusted' in kwargs['input'][0]['content']


def test_dashboard_filtering_does_not_call_pipeline():
    result, _, _ = fixture(evaluate=lambda *args: decision())
    with patch('research.run_research', return_value=result) as pipeline:
        app = AppTest.from_file('../app.py', default_timeout=15).run()
        app.checkbox[0].check().run()
        app.button[0].click().run()
        app.selectbox[0].select('negative').run()
        assert pipeline.call_count == 1
    assert not app.exception
    assert 'Additional Investigation' in [h.value for h in app.subheader]
    text = [element.value for element in app.markdown]
    assert 'Selected theme: reliability' in text
    assert any('Insufficient examples' in value for value in text)
    assert any('Accepted targeted candidates: 1' in value for value in text)
    assert any('confirmed new supporting issues: 1' in value for value in text)
    links = [link for link in app.get('link_button')
             if link.proto.label.startswith('Supporting GitHub issue')]
    assert len(links) == 1
    assert links[0].proto.url == result.investigation.supporting[0].url
    assert result.investigation.supporting[0].title in [element.value for element in app.text]


def test_shared_rate_limit_state_blocks_targeted_search():
    from research import RateAwareClient
    original = RateAwareClient.get
    def get(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        if self.requests == 12:
            self.ready = self.deadline + 1
        return response
    with patch.object(RateAwareClient, 'get', get):
        result, requests, batches = fixture(evaluate=lambda *args: decision())
    assert len(requests) == 12 and len(batches) == 5
    assert result.investigation.status == 'failed'
    assert result.investigation.github_requests == 0
    assert result.counts['collected'] == 50


def test_evaluation_failure_preserves_baseline():
    def failure(*args):
        raise ResearchError('Synthetic evaluator failure')
    result, requests, batches = fixture(evaluate=failure)
    assert len(requests) == 12 and len(batches) == 5
    assert result.investigation.status == 'skipped'
    assert result.counts['collected'] == 50


@pytest.mark.parametrize("kind", ["uncertain", "wrong_theme"])
def test_irrelevant_or_wrong_theme_is_not_supporting(monkeypatch, kind):
    original = Label
    def make_label(**kwargs):
        if kwargs['source_id'].endswith(':999'):
            if kind == 'uncertain':
                kwargs['relevance'] = 'uncertain'
            else:
                kwargs['complaint_themes'] = ['documentation']
        return original(**kwargs)
    monkeypatch.setitem(globals(), 'Label', make_label)
    result, _, _ = fixture(evaluate=lambda *args: decision())
    assert not result.investigation.supporting


def test_no_relevant_evidence_skips_evaluation(monkeypatch):
    original = Label
    def irrelevant(**kwargs):
        return original(**{**kwargs, 'relevance': 'irrelevant'})
    monkeypatch.setitem(globals(), 'Label', irrelevant)
    def forbidden(*args):
        pytest.fail('No evaluator without relevant evidence')
    result, requests, _ = fixture(evaluate=forbidden)
    assert len(requests) == 12 and result.investigation.model_calls == 5
    assert result.investigation.status == 'skipped'


def test_hard_request_cap_prevents_fourteenth_call():
    from research import RateAwareClient
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'items': []})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        paced = RateAwareClient(client, lambda msg: None, max_requests=13)
        for _ in range(13):
            paced.get('https://api.github.com/search/issues')
        with pytest.raises(ResearchError):
            paced.get('https://api.github.com/search/issues')
    assert len(calls) == 13


def test_recent_path_retains_small_baseline():
    requests, batches = [], []
    def handler(request):
        requests.append(request)
        assert request.url.params['per_page'] == '10'
        return httpx.Response(200, json={'items': [raw(i) for i in range(1, 11)],
                                        'total_count': 100, 'incomplete_results': False})
    def classify(question, items):
        batches.append(items)
        return [Label(source_id=i.source_id, relevance='relevant', sentiment='neutral',
                      complaint_themes=['reliability']) for i in items]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = run_research('reliability', REPO, START, END, client=client, classify=classify,
            investigate=True, evaluate=lambda *args: {
                'investigate': False, 'theme': None, 'reason': 'sufficient_evidence', 'source_ids': []})
    assert len(requests) == len(batches) == 1
    assert len(result.batch.items) == 10 and result.monthly.empty
    assert result.investigation.model_calls == 2


@pytest.mark.parametrize('status', [403, 429])
def test_targeted_rate_limit_preserves_baseline_without_retry(status):
    original = httpx.MockTransport.handle_request
    def handle(self, request):
        if 'in:title,body' in request.url.params['q']:
            return httpx.Response(status, headers={'retry-after': '60',
                'x-ratelimit-remaining': '0', 'x-ratelimit-reset': '9999999999'})
        return original(self, request)
    with patch.object(httpx.MockTransport, 'handle_request', handle):
        result, requests, batches = fixture(evaluate=lambda *args: decision())
    assert len(requests) == 12  # The intercepted targeted response bypasses the fixture handler.
    assert result.investigation.github_requests == 1
    assert result.investigation.model_calls == 6
    assert len(batches) == 5
    assert result.investigation.status == 'failed'
    assert result.counts['collected'] == result.counts['classified'] == 50
    assert not result.investigation.supporting
    assert result.investigation.limitations
