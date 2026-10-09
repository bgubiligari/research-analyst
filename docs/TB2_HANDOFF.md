# TB2 Handoff — First Historical Sentiment Slice

## Status and scope

Implemented on `feature/tb2-historical-sampling`; verification and checkpointing
approved on 2026-10-08. TB1 remains available. Monthly complaint trends and TB3
are not approved. Do not merge or push without explicit authorization.

## Implemented workflow

The existing UI-called `run_research` supports historical retrieval → existing
structured OpenAI classifier → deterministic Pandas monthly metrics → one
historical sentiment chart. The dashboard includes monthly coverage, interval
errors, original GitHub links and in-memory evidence filters. Monthly columns
have help descriptions. Two generic yellow sampling banners were removed at the
user’s request; specific partial/sparse/incomplete warnings remain visible.

## Sampling and bounds

- Default: last six completed UTC calendar months. October 8, 2026 gives April
  1–September 30. Custom ranges support one to six buckets; partial months are marked.
- Allocate 50 issues evenly oldest-first: six months receive 9,9,8,8,8,8.
- Split each clipped month into contiguous disjoint date halves. Odd days and
  quotas go early. Early queries sort creation ascending; late queries descending.
  One-day ranges use one request. No pagination, randomness, retries, refill or redistribution.
- Query only public issues in the selected repository/date interval, without
  topic keywords. Reject PRs, invalid URLs/timestamps and duplicate identities.
- At most 12 sequential searches, 50 accepted unique issues and five model
  batches of at most ten. Failed model batches remain unclassified.
- Rate-limit remaining/reset and Retry-After headers pace requests within a
  180-second retrieval budget. Wait chunks are at most 30 seconds. Budget stops
  and 403/429 preserve prior results and mark remaining intervals unattempted.

## Analytics and architecture

Sentiment N = relevant classified issues excluding unclear sentiment. Zero N
produces N/A; N < 5 retains table counts/percentages but suppresses chart points.
Explicit series segments prevent connecting gaps. Chart data derives from the
same monthly Pandas table. Match totals, returned/rejected/accepted records,
classification exclusions and failures remain separate; unknown totals are N/A.
TB1 aggregate multi-label complaint metrics are unchanged.

One local synchronous Streamlit/Python process managed by uv. `app.py` renders;
`research.py` contains retrieval, typed classification, metrics and helpers.
`tests/test_historical.py` adds synthetic production-path tests alongside the
32 existing TB1 tests. No additional dependencies or infrastructure were added.
Source text and labels stay in session memory; no persistence or source logging.
Credentials are server-side in ignored `.env`; OpenAI requests use `store=False`.

## Validation

- `uv run pytest -q`: 42 passed at the final checkpoint.
- `uv run ruff check .`: passed.
- Staged whitespace and sensitive-data inspection performed before commit.
- Tests cover calendar/year/leap boundaries, quotas, caps, UTC bucketing,
  rejected identities, rate pacing/stops, classification failures/exclusions,
  exact 2/5 = 40% arithmetic, sparse gaps, chart/table equality and UI filtering.
- Verification-only oversized synthetic responses respected 50/12/5 caps.
- Fresh live classification with the current prompt was performed before approval.
  Latest browser run: 1,112 reported matches; 50 retrieved/classified; 44 relevant,
  6 excluded; all 12 searches successful. A separate earlier live run found 46
  relevant, illustrating model variability. Neither run is an accuracy benchmark.
- Real browser: Research submission, chart/table rendering, evidence filtering
  (26 negative rows; 18 with reliability), inspector/source destination and
  TB1/TB2 switching checked. No console errors observed. Tooltip popup itself
  remains unverified through automation. No fresh live run needed for this
  documentation-only checkpoint.

## Known limitations and next review

The sample favors monthly edges and is not representative; equal timestamp
ordering depends on GitHub. Sparse relevance, API limits and unused quotas can
reduce coverage. Text truncation and model interpretation can affect labels.
Constructive contributions may be interpreted as positive; no prompt change was
made after the user’s review. Provider retention is not guaranteed by `store=False`.
HTTP timeouts are per network phase; the retrieval budget is checked before
calls/waits rather than enforcing a hard whole-run deadline. Refresh/disconnect
can lose session evidence. Do not save raw sources or item-level labels.

Stop after checkpointing. Any further TB2 work or TB3 requires explicit approval.
