"""Synthetic Quick/Deep fixtures through the same production pipeline as the UI."""
from datetime import date
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from research import (
    GapDecision,
    Label,
    RateAwareClient,
    historical_chart_data,
    run_research,
)

START, END = date(2026, 4, 1), date(2026, 9, 30)
REPO = 'synthetic/repo'


def run_fixture(historical, deep, gap=False):
    requests, models, evaluations, ceilings = [], [], [], []
    def handler(request):
        requests.append(request)
        targeted = 'in:title,body' in request.url.params['q']
        first = request.url.params['q'].split('created:')[1].split('..')[0]
        numbers = [1, 9000] if targeted else range(len(requests)*100,
            len(requests)*100 + int(request.url.params['per_page']))
        if targeted:
            numbers = [100, 9000]  # One baseline duplicate and one new record.
        records = [{'number': n, 'html_url': f'https://github.com/{REPO}/issues/{n}',
            'created_at': first+'T12:00:00Z', 'title': 'Synthetic fixture',
            'body': 'Synthetic reliability complaint'} for n in numbers]
        return httpx.Response(200, json={'items': records, 'total_count': 100,
                                        'incomplete_results': False})
    def classify(question, items):
        models.append([i.source_id for i in items])
        return [Label(source_id=i.source_id, relevance='relevant', sentiment='neutral',
                      complaint_themes=['reliability']) for i in items]
    def evaluate(*args):
        evaluations.append(args)
        return GapDecision(investigate=gap, theme='reliability' if gap else None,
            reason='insufficient_examples' if gap else 'sufficient_evidence', source_ids=[])
    original = RateAwareClient.__init__
    def init(self, *args, **kwargs):
        ceilings.append(kwargs['max_requests'])
        original(self, *args, **kwargs)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, patch.object(
            RateAwareClient, '__init__', init):
        result = run_research('Explain reliability complaints', REPO, START, END,
            historical=historical, investigate=deep, evaluate=evaluate, classify=classify, client=client, narrate=None)
    return result, requests, models, evaluations, ceilings


@pytest.mark.parametrize('historical', [False, True])
@pytest.mark.parametrize('deep,gap', [(False, False), (True, False), (True, True)])
def test_all_sampling_depth_paths_and_budgets(historical, deep, gap):
    result, requests, models, evaluations, ceilings = run_fixture(historical, deep, gap)
    base_requests, base_models = (12, 5) if historical else (1, 1)
    assert len(result.batch.items) == (50 if historical else 10)
    assert ceilings == [base_requests + int(deep)]
    assert len(requests) == base_requests + int(deep and gap)
    assert len(evaluations) == int(deep)
    assert len(models) == base_models + int(deep and gap)
    assert result.investigation.model_calls == base_models + int(deep) + int(deep and gap)
    assert result.investigation.github_requests == int(deep and gap)
    assert result.investigation.status == ('complete' if deep and gap else 'skipped' if deep else 'disabled')
    if deep and gap:
        assert result.investigation.duplicates == 1
        assert models[-1] == ['github:synthetic/repo:9000']
        assert result.investigation.supporting[0] in result.investigation.batch.items
    baseline = run_fixture(historical, False)[0]
    assert result.counts == baseline.counts and result.labels == baseline.labels
    for name in ('sentiment', 'complaints', 'monthly'):
        pd.testing.assert_frame_equal(getattr(result, name), getattr(baseline, name))
    if historical:
        pd.testing.assert_frame_equal(historical_chart_data(result.monthly),
                                     historical_chart_data(baseline.monthly))


@pytest.mark.parametrize('historical', [False, True])
@pytest.mark.parametrize('deep', [False, True])
def test_executed_configuration_survives_controls_and_filters(historical, deep):
    result = run_fixture(historical, deep, gap=deep)[0]
    sampling, depth = ('Historical' if historical else 'Recent'), ('Deep' if deep else 'Quick')
    with patch('research.run_research', return_value=result) as pipeline:
        app = AppTest.from_file('../app.py', default_timeout=15).run()
        app.radio[0].set_value('Historical (TB2)' if historical else 'Recent (TB1)').run()
        app.radio[1].set_value(depth).run()
        assert pipeline.call_count == 0
        app.button[0].click().run()
        assert pipeline.call_args.kwargs['historical'] is historical
        assert pipeline.call_args.kwargs['investigate'] is deep
        app.radio[0].set_value('Recent (TB1)' if historical else 'Historical (TB2)').run()
        app.radio[1].set_value('Quick' if deep else 'Deep').run()
        app.selectbox[0].select('neutral').run()
        app.selectbox[1].select('reliability').run()
        app.selectbox[2].select('All').run()
        app.selectbox[2].select('Uncertain').run()
        assert pipeline.call_count == 1
    assert not app.exception
    assert f'Executed research: {sampling} {depth}' in [m.value for m in app.markdown]
    assert ('Additional Investigation' in [h.value for h in app.subheader]) is deep
    assert ('Historical Sentiment in Sampled GitHub Issues' in [h.value for h in app.subheader]) is historical
