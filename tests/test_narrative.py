"""Synthetic TB4-A fixtures use the production pipeline, never live source text."""
import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from research import (
    Label,
    NarrativeOutput,
    NarrativeResult,
    ResearchError,
    generate_narrative,
    historical_chart_data,
    narrative_context,
    narrative_metrics,
    run_research,
    validate_narrative,
)

START, END = date(2026, 4, 1), date(2026, 9, 30)
REPO = 'synthetic/narrative'


def claim(evidence_id, **changes):
    return dict(section='key_finding', text='Reports describe unexpected execution failures.',
        theme='reliability', metric_keys=['complaints.reliability'],
        evidence=[{'evidence_id': evidence_id, 'role': 'supporting'}], **changes)


def valid(context):
    evidence_id = next(k for k, v in context['evidence'].items() if v['sentiment'] == 'negative')
    return {'status': 'supported', 'claims': [claim(evidence_id)]}, {
        'input_tokens': 4000, 'output_tokens': 100, 'total_tokens': 4100,
        'raw_secret': 'must not survive'}


def fixture(historical=False, deep=False, gap=False, narrator=valid, clock=None):
    requests, classifications, evaluations, narratives = [], [], [], []
    def handler(request):
        requests.append(request)
        first = request.url.params['q'].split('created:')[1].split('..')[0]
        targeted = 'in:title,body' in request.url.params['q']
        numbers = [100, 9000] if targeted else range(len(requests)*100,
            len(requests)*100+int(request.url.params['per_page']))
        items = [{'number': n, 'html_url': f'https://github.com/{REPO}/issues/{n}',
                  'title': 'Synthetic failure complaint', 'body': 'Synthetic failure during execution.',
                  'created_at': first+'T12:00:00Z'} for n in numbers]
        return httpx.Response(200, json={'items': items, 'total_count': 200,
                                        'incomplete_results': True})
    def classify(question, items):
        classifications.append([i.source_id for i in items])
        return [Label(source_id=i.source_id, relevance='relevant',
            sentiment='positive' if i.number % 100 == 1 else 'negative',
            complaint_themes=['reliability']) for i in items]
    def evaluate(*args):
        evaluations.append(args)
        return {'investigate': gap, 'theme': 'reliability' if gap else None,
                'reason': 'missing_concrete_detail' if gap else 'sufficient_evidence', 'source_ids': []}
    def narrate(context):
        narratives.append(context)
        return narrator(context)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = run_research('Explain reliability complaints', REPO, START, END,
            historical=historical, investigate=deep, classify=classify, evaluate=evaluate,
            narrate=narrate if narrator is not None else None, client=client,
            **({'monotonic': clock} if clock else {}))
    return result, requests, classifications, evaluations, narratives


@pytest.mark.parametrize('historical', [False, True])
@pytest.mark.parametrize('deep,gap', [(False, False), (True, False), (True, True)])
def test_production_paths_caps_and_baseline_integrity(historical, deep, gap):
    result, requests, classifications, evaluations, narratives = fixture(historical, deep, gap)
    baseline = fixture(historical, narrator=None)[0]
    base = 5 if historical else 1
    assert len(result.batch.items) == (50 if historical else 10)
    assert len(requests) == (12 if historical else 1)+int(deep and gap)
    assert len(classifications) == base+int(deep and gap)
    assert len(evaluations) == int(deep) and len(narratives) == 1
    assert result.investigation.model_calls == base+int(deep)+int(deep and gap)+1
    assert result.investigation.model_calls <= base+2*int(deep)+1
    assert result.narrative.status == 'available'
    assert result.narrative.usage == {'input_tokens': 4000, 'output_tokens': 100, 'total_tokens': 4100}
    assert result.labels == baseline.labels and result.counts == baseline.counts
    for name in ('sentiment', 'complaints', 'monthly'):
        pd.testing.assert_frame_equal(getattr(result, name), getattr(baseline, name))
    if historical:
        pd.testing.assert_frame_equal(historical_chart_data(result.monthly), historical_chart_data(baseline.monthly))
    if gap:
        assert classifications[-1] == [f'github:{REPO}:9000']
        assert result.investigation.duplicates == 1
    refs = result.narrative.claims[0]['citations']
    assert refs[0]['url'] == result.narrative.evidence[refs[0]['evidence_id']]['url']
    assert refs[0]['source_id'] in {i.source_id for i in result.batch.items}
    assert result.narrative.claims[0]['metrics'] == [
        f'Sampled baseline reliability: {len(result.batch.items)} of {len(result.batch.items)} eligible issues (100.0%).']


@pytest.mark.parametrize('change', [
    {'evidence': [{'evidence_id': 'invented', 'role': 'supporting'}]},
    {'metric_keys': ['complaints.invented']},
    {'metric_keys': ['complaints.performance']},
    {'text': 'Twenty percent of users report failures.'},
    {'text': 'Failure rate is 25%.'},
    {'text': 'See https://example.com.'},
    {'text': 'The root cause is adoption.'},
    {'text': 'Most developers experience failures.'},
    {'evidence': []},
    {'theme': 'performance'},
    {'section': 'unknown'},
])
def test_invalid_claim_filtered_without_discarding_valid_sibling(change):
    def narrator(context):
        key = next(k for k, v in context['evidence'].items() if v['sentiment'] == 'negative')
        invalid = claim(key)
        invalid.update(change)
        return {'status': 'supported', 'claims': [invalid, claim(key)]}, {}
    result = fixture(narrator=narrator)[0]
    assert result.narrative.rejected == 1
    assert len(result.narrative.claims) == 1 and result.narrative.status == 'available'


@pytest.mark.parametrize('failure', ['api', 'refusal', 'malformed', 'rejected', 'insufficient'])
def test_failure_preserves_results_and_dashboard(failure):
    def narrator(context):
        if failure == 'api':
            raise ResearchError('Synthetic provider failure')
        if failure == 'refusal':
            return None, {}
        if failure == 'malformed':
            return {'status': 'unknown'}, {}
        if failure == 'insufficient':
            return {'status': 'insufficient_evidence', 'claims': []}, {}
        return {'status': 'supported', 'claims': [claim('invented')]}, {}
    result = fixture(narrator=narrator)[0]
    baseline = fixture(narrator=None)[0]
    assert result.narrative.status in ('unavailable', 'insufficient_evidence')
    assert result.counts == baseline.counts and result.sentiment.equals(baseline.sentiment)
    assert result.investigation.model_calls == 2
    with patch('research.run_research', return_value=result):
        app = AppTest.from_file('../app.py', default_timeout=15).run()
        app.button[0].click().run()
    assert not app.exception
    assert any('Narrative unavailable' in info.value for info in app.info)
    assert 'Research Findings' in [h.value for h in app.subheader]


@pytest.mark.parametrize('ns,comparisons,patterns', [
    ([4, 8, 4], 0, 0), ([5, 5, 4], 4, 0), ([5, 5, 5], 8, 4),
    ([5, 0, 5], 0, 0), ([5, 5, 0, 5, 5], 8, 0),
])
def test_historical_eligibility_and_gaps(ns, comparisons, patterns):
    result = fixture(historical=True, narrator=None)[0]
    result.monthly = result.monthly.iloc[:len(ns)].copy()
    result.monthly['denominator'] = ns
    metrics = narrative_metrics(result)
    assert sum(k.startswith('comparison.') for k in metrics) == comparisons
    assert sum(k.startswith('pattern.') for k in metrics) == patterns
    assert all('population trend' in m['text'] for m in metrics.values() if m['kind'] == 'pattern')
    historical = {'section': 'historical', 'text': 'Invented direction 99%', 'theme': None,
                  'metric_keys': ['comparison.2026-04.2026-05.negative'], 'evidence': []}
    narrative = NarrativeResult()
    validate_narrative({'status': 'supported', 'claims': [historical]}, narrative, metrics)
    if ns[:2] == [5, 5]:
        assert narrative.claims[0]['text'] == ''  # Never display model-authored directions/numbers.
        assert narrative.claims[0]['metrics'] == [metrics[historical['metric_keys'][0]]['text']]
    else:
        assert narrative.rejected == 1 and not narrative.claims


def test_exact_comparison_arithmetic_and_no_bridge():
    result = fixture(historical=True, narrator=None)[0]
    result.monthly.loc[0, ['negative', 'denominator', 'negative_percent']] = [2, 5, 40]
    result.monthly.loc[1, ['negative', 'denominator', 'negative_percent']] = [3, 5, 60]
    result.monthly.loc[2, 'denominator'] = 0
    metrics = narrative_metrics(result)
    comparison = metrics['comparison.2026-04.2026-05.negative']
    assert comparison['difference'] == 20
    assert '2/5 (40.0%)' in comparison['text'] and '3/5 (60.0%)' in comparison['text']
    assert '+20.0 percentage points' in comparison['text']
    assert not any(k.startswith('comparison.2026-05.') for k in metrics)


def test_context_bounded_stable_ids_contrast_and_targeted_rules():
    result = fixture(historical=True, deep=True, gap=True, narrator=None)[0]
    context, manifest, metrics = narrative_context('Synthetic', REPO, START, END, result, True, True)
    assert len([r for r in manifest.values() if r['cohort'] == 'baseline']) == 6
    assert len([r for r in manifest.values() if r['cohort'] == 'targeted']) == 1
    assert len(json.dumps(context, ensure_ascii=False).encode()) <= 24000
    assert all('url' not in r for r in context['evidence'].values())
    assert all('source_id' not in r for r in context['evidence'].values())
    assert any('contrasting' in warning for warning in context['limitations'])
    assert manifest == narrative_context('Synthetic', REPO, START, END, result, True, True)[1]
    positive = next(k for k, v in manifest.items() if v['sentiment'] == 'positive')
    negative = next(k for k, v in manifest.items() if v['sentiment'] == 'negative')
    targeted = next(k for k, v in manifest.items() if v['cohort'] == 'targeted')
    draft = claim(negative)
    draft['evidence'].append({'evidence_id': positive, 'role': 'contrasting'})
    targeted_claim = claim(targeted)
    narrative = NarrativeResult(evidence=manifest)
    validate_narrative({'status': 'supported', 'claims': [draft, targeted_claim]}, narrative, metrics)
    assert narrative.rejected == 1 and len(narrative.claims) == 1
    assert 'contrasting' in narrative.claims[0]['text']
    targeted_claim['metric_keys'] = []
    accepted = NarrativeResult(evidence=manifest)
    validate_narrative({'status': 'supported', 'claims': [targeted_claim]}, accepted, metrics)
    assert accepted.claims[0]['citations'][0]['cohort'] == 'targeted'


def test_dashboard_filters_and_modes_never_regenerate_narrative():
    result = fixture(historical=True, deep=True, gap=True)[0]
    with patch('research.run_research', return_value=result) as pipeline:
        app = AppTest.from_file('../app.py', default_timeout=15).run()
        app.radio[1].set_value('Deep').run()
        app.button[0].click().run()
        app.radio[0].set_value('Recent (TB1)').run()
        app.radio[1].set_value('Quick').run()
        app.selectbox[0].select('negative').run()
        app.selectbox[2].select('All').run()
        assert pipeline.call_count == 1
    assert not app.exception
    assert 'Executed research: Historical Deep' in [m.value for m in app.markdown]
    assert 'Research Summary' in [h.value for h in app.subheader]
    assert any('Summary reuses the validated findings' in c.value for c in app.caption)
    assert any(result.narrative.claims[0]['text'] == element.value for element in app.text)


def test_sdk_boundary_usage_no_retries_or_store(monkeypatch):
    recorded = {}
    parsed = NarrativeOutput(status='insufficient_evidence', claims=[])
    class FakeOpenAI:
        def __init__(self, **kwargs):
            recorded['client'] = kwargs
            self.responses = self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def parse(self, **kwargs):
            recorded['request'] = kwargs
            return SimpleNamespace(output_parsed=parsed, usage=SimpleNamespace(model_dump=lambda: {
                'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30}))
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-test-key')
    monkeypatch.setenv('OPENAI_MODEL', 'synthetic-test-model')
    monkeypatch.setattr('research.OpenAI', FakeOpenAI)
    output, usage = generate_narrative({'question': 'Synthetic'}, timeout=7)
    assert output == parsed and usage['total_tokens'] == 30
    assert recorded['client'] == {'timeout': 7, 'max_retries': 0}
    assert recorded['request']['store'] is False and recorded['request']['max_output_tokens'] == 2000
    assert recorded['request']['text_format'] is NarrativeOutput


def test_execution_deadline_skips_narrative():
    ticks = [0]
    def clock():
        ticks[0] += 50
        return ticks[0]
    result, _, _, _, calls = fixture(historical=True, clock=clock)
    assert not calls
    assert result.narrative.status == 'unavailable'
    assert result.investigation.model_calls <= 6


def test_empty_and_irrelevant_evidence_skip_narrative_without_calls():
    from research import Issue, SearchBatch
    issue = Issue(source_id=f'github:{REPO}:1', number=1,
        url=f'https://github.com/{REPO}/issues/1', created_at='2026-04-01T00:00:00Z',
        title='Synthetic unrelated record', body='Synthetic unrelated evidence')
    for items in ([], [issue]):
        calls = []
        result = run_research('Synthetic', REPO, START, END,
            fetch=lambda *args, items=items: SearchBatch(items, len(items), False),
            classify=lambda *args: [Label(source_id=issue.source_id, relevance='irrelevant',
                                         sentiment='unclear', complaint_themes=[])],
            narrate=lambda context, calls=calls: calls.append(context))
        assert not calls and result.narrative.status == 'unavailable'
        assert not narrative_metrics(result)
        assert result.sentiment.percent.isna().all()


def test_multibyte_context_and_sparse_disclosure():
    result = fixture(narrator=None)[0]
    result.batch.items = result.batch.items[:3]
    result.labels = result.labels[:3]
    for issue in result.batch.items:
        issue.title = 'Synthetic '+ '\U0001f600'*200
        issue.body = 'Synthetic failure '+ '\U0001f600'*5500
    context, manifest, _ = narrative_context('Synthetic', REPO, START, END, result, False, False)
    assert len(json.dumps(context, ensure_ascii=False).encode()) <= 24000
    assert len(manifest) <= 3
    assert any('Sparse relevant baseline' in warning for warning in context['limitations'])
    assert all(len(record['excerpt']) <= 1200 for record in manifest.values())


def test_narrative_call_reserved_without_broadening_classification_budget():
    from research import Issue, SearchBatch
    items = [Issue(source_id=f'github:{REPO}:{n}', number=n,
        url=f'https://github.com/{REPO}/issues/{n}', created_at='2026-04-01T00:00:00Z',
        title='Synthetic fixture', body='Synthetic failure') for n in range(1, 31)]
    classified = []
    def classify(question, group):
        classified.append(group)
        return [Label(source_id=i.source_id, relevance='relevant', sentiment='negative',
                      complaint_themes=['reliability']) for i in group]
    result = run_research('Synthetic', REPO, START, END,
        fetch=lambda *args: SearchBatch(items, 30, False), classify=classify, narrate=valid)
    # The intentionally oversized injected boundary must not spend narrative capacity on classification.
    assert len(classified) == 1
    assert result.investigation.model_calls == 2 and result.narrative.status == 'available'
    assert result.counts['unclassified'] == 20


def test_aggregate_provenance_strips_issue_links_and_combined_keeps_both():
    def narrator(context):
        key = next(iter(context['evidence']))
        monthly = next(k for k in context['metrics'] if k.startswith('pattern.'))
        historical = claim(key)
        historical.update(section='historical', theme=None, metric_keys=[monthly])
        aggregate = claim(key)
        aggregate.update(text='')
        return {'status': 'supported', 'claims': [historical, aggregate, claim(key)]}, {}
    result = fixture(historical=True, narrator=narrator)[0]
    historical, aggregate, combined = result.narrative.claims
    assert historical['support_kind'] == aggregate['support_kind'] == 'aggregate'
    assert not historical['citations'] and not aggregate['citations']
    assert historical['metric_keys'][0].startswith('pattern.')
    assert aggregate['metric_keys'] == ['complaints.reliability']
    assert combined['support_kind'] == 'combined'
    assert combined['metric_keys'] == ['complaints.reliability']
    assert combined['citations'][0]['source_id'] in {i.source_id for i in result.batch.items}
    cited = next(i for i in result.batch.items if i.source_id == combined['citations'][0]['source_id'])
    assert combined['citations'][0]['url'] == cited.url


@pytest.mark.parametrize('changes,code', [
    ({'evidence': [{'evidence_id': 'synthetic:missing', 'role': 'supporting'}]}, 'UNKNOWN_EVIDENCE_ID'),
    ({'metric_keys': ['synthetic.missing']}, 'UNKNOWN_METRIC_KEY'),
    ({'evidence': [], 'metric_keys': []}, 'MISSING_REQUIRED_SUPPORT'),
    ({'theme': 'performance'}, 'INVALID_THEME'),
    ({'theme': 'invented'}, 'INVALID_THEME'),
    ({'section': 'historical'}, 'UNSUPPORTED_HISTORICAL_CLAIM'),
    ({'section': 'invalid'}, 'INVALID_CLAIM_STRUCTURE'),
    ({'text': 'Most developers experience failures.'}, 'UNSUPPORTED_CLAIM'),
])
def test_specific_session_diagnostics_preserve_valid_sibling(changes, code):
    def narrator(context):
        key = next(iter(context['evidence']))
        invalid = claim(key)
        invalid.update(changes)
        return {'status': 'supported', 'claims': [invalid, claim(key)]}, {}
    result = fixture(narrator=narrator)[0]
    assert result.narrative.status == 'available' and len(result.narrative.claims) == 1
    diagnostic, = result.narrative.diagnostics
    assert diagnostic['claim_index'] == 1
    assert diagnostic['reason_code'] == code and diagnostic['explanation']
    assert diagnostic['section'] == changes.get('section', 'key_finding')
    assert diagnostic['candidate_text'] == changes.get('text', claim('synthetic')['text'])
    assert set(diagnostic) == {'claim_index', 'section', 'reason_code', 'explanation',
                               'evidence_ids', 'metric_keys', 'candidate_text'}


def test_qualitative_support_and_feature_request_not_confirmed_failure():
    result = fixture(narrator=None)[0]
    for issue in result.batch.items:
        issue.issue_type = 'feature_request'
    context, manifest, metrics = narrative_context('Synthetic', REPO, START, END, result, False, False)
    key = next(iter(manifest))
    assert context['evidence'][key]['issue_type'] == 'feature_request'
    bad = claim(key)
    bad.update(text='Reports establish a confirmed failure in production.', metric_keys=[])
    good = claim(key)
    good.update(text='Developers request additional provider compatibility.', metric_keys=[])
    narrative = NarrativeResult(evidence=manifest)
    validate_narrative({'status': 'supported', 'claims': [bad, good]}, narrative, metrics)
    assert narrative.diagnostics[0]['reason_code'] == 'UNSUPPORTED_CLAIM'
    assert narrative.claims[0]['support_kind'] == 'qualitative'
    assert narrative.claims[0]['citations'][0]['issue_type'] == 'feature_request'
    manifest[key]['relevance'] = 'irrelevant'
    rejected = NarrativeResult(evidence=manifest)
    validate_narrative({'status': 'supported', 'claims': [good]}, rejected, metrics)
    assert rejected.diagnostics[0]['reason_code'] == 'MISSING_REQUIRED_SUPPORT'


def test_diagnostics_are_session_only_not_saved_or_logged(tmp_path, monkeypatch, caplog, capsys):
    monkeypatch.chdir(tmp_path)
    def narrator(context):
        key = next(iter(context['evidence']))
        invalid = claim(key)
        invalid.update(evidence=[], text='Synthetic rejected candidate')
        # Unapproved fields must never enter the diagnostic record.
        invalid['raw_response'] = 'Synthetic raw source canary'
        return {'status': 'supported', 'claims': [invalid, claim(key)]}, {}
    result = fixture(narrator=narrator)[0]
    diagnostic, = result.narrative.diagnostics
    assert 'raw_response' not in diagnostic and 'excerpt' not in diagnostic and 'body' not in diagnostic
    assert not list(tmp_path.iterdir())
    assert not caplog.text and not capsys.readouterr().out


def test_dashboard_debug_expander_and_aggregate_without_issue_citation():
    def narrator(context):
        key = next(iter(context['evidence']))
        historical = claim(key)
        historical.update(section='historical', theme=None,
                          metric_keys=[next(k for k in context['metrics'] if k.startswith('pattern.'))])
        invalid = claim('synthetic:missing')
        return {'status': 'supported', 'claims': [historical, invalid, claim(key)]}, {}
    result = fixture(historical=True, narrator=narrator)[0]
    with patch('research.run_research', return_value=result) as pipeline:
        app = AppTest.from_file('../app.py', default_timeout=15).run()
        app.button[0].click().run()
        assert any(e.label == 'Developer debug: narrative rejection diagnostics' for e in app.expander)
        assert any('UNKNOWN_EVIDENCE_ID' in t.value for t in app.text)
        assert any('Metric provenance: pattern.' in c.value for c in app.caption)
        app.radio[1].set_value('Deep').run()
        assert pipeline.call_count == 1
        assert not app.exception
    # Starting a failed replacement run clears the prior active evidence and diagnostics.
    with patch('research.run_research', side_effect=[result, ResearchError('Synthetic retrieval failure')]):
        app = AppTest.from_file('../app.py', default_timeout=15).run()
        app.button[0].click().run()
        assert 'active_result' in app.session_state
        app.button[0].click().run()
        assert 'active_result' not in app.session_state
        assert not any(e.label == 'Developer debug: narrative rejection diagnostics' for e in app.expander)
        assert not app.exception


def test_github_source_types_do_not_change_accepted_identity_or_count():
    from research import search_issues
    types = [{'name': 'Bug'}, {'name': 'Feature'}, None, {'name': 'Task'}, 'unrecognized']
    def handler(request):
        return httpx.Response(200, json={'total_count': len(types), 'incomplete_results': False,
            'items': [{'number': n, 'html_url': f'https://github.com/{REPO}/issues/{n}',
                       'created_at': '2026-04-01T12:00:00Z', 'title': 'Synthetic source',
                       'body': 'Synthetic source', 'type': source_type}
                      for n, source_type in enumerate(types, start=1)]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        batch = search_issues(REPO, START, END, client=client)
    assert batch.rejected == 0 and len(batch.items) == len(types)
    assert [i.number for i in batch.items] == list(range(1, 6))
    assert [i.issue_type for i in batch.items] == [
        'reported_bug', 'feature_request', 'unspecified', 'unspecified', 'unspecified']
