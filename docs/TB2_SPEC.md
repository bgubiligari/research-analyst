# TB2 Specification — Historical Sampling and Sentiment Trends

**Status:** First sentiment-only slice completed; verification approved and checkpoint authorized; monthly complaint trends and later bullets remain unapproved. This document covers TB2 only. Read AGENTS.md and TB1_HANDOFF.md before proposing an implementation plan.

## A. Goal

Support questions such as “How have developer complaints about LangChain changed over the past six months?” Extend the working pipeline with calendar-month coverage rather than a single newest-first sample. Results describe sampled GitHub Issues, not a statistically representative population.

## B. Historical Sampling

- Split the selected date range into non-overlapping UTC calendar-month windows, clipping first/last windows to requested dates. Default: last six completed UTC calendar months (October 8, 2026 → April–September). Custom ranges support one to six buckets; disclose partial months.
- Fixed evenly allocated capacity constrained by the **50-issue total TB2 budget**. Targets are not guarantees. Allocate capacity evenly across windows before retrieval; for six windows, 8 each plus one extra for the first two chronologically. Never silently increase the global cap to meet monthly targets.
- Baseline queries retain repository, public issue and creation-date constraints; exclude PRs. Do not add negative keywords or question-aware retrieval in TB2.
- First slice: split each clipped monthly interval into disjoint date halves (odd day goes early). Early half uses ascending creation order; late half descending. Odd quota goes early. No random sampling, pagination, quota redistribution or refill. Selection favors the edges within each month; GitHub may order identical timestamps without a guaranteed tie-break.
- Deduplicate globally using canonical repository/issue identifiers and validate original URLs. Preserve issue `created_at` separately from run retrieval timestamp; bucket only by creation time.
- Report GitHub match counts separately from actual accepted sampled counts. Unknown totals are N/A, never zero. Retain incomplete flags, errors, discarded counts, quota limits and strategy per month.
- Bounds: at most two requests per month and 12 sequential requests total; no retries or pagination. A 180-second retrieval budget includes pacing and requests. Rate-limit remaining/reset and Retry-After headers govern waiting in chunks of at most 30 seconds; stop if waiting exceeds the budget. A 403/429 stops further requests without retry. Preserve partial results, error and unattempted interval metadata. Retain request timeouts of at most 20 seconds.
- Classify accepted issues in batches of at most 10 using the existing engine, at most five calls for 50 issues; failed batches stay unclassified. Keep everything synchronous and in memory.

## C. Monthly Sentiment Analytics

For every requested month, including empty/error months, compute reported matches where available, retrieved count, relevant classified count, positive/negative/neutral/mixed counts and percentages, unclear count, unclassified count, and sentiment denominator. Also expose irrelevant/uncertain exclusions.

Reuse TB1: sentiment denominator = relevant classified issues with sentiment other than unclear. No classifiable evidence means N/A/insufficient data, not 0% negative. For denominator below 5, retain counts and warn; suppress percentage trend points. Never connect a line across missing/suppressed months. All arithmetic is Python/Pandas.

## D. Monthly Complaint Analytics — deferred, not implemented

Compute theme counts, prevalence, deterministic top categories, and adjacent-month prevalence differences in percentage points. Denominator = all relevant classified issues, including empty themes and unclear sentiment. Count an issue once per theme; allow overlapping categories and totals above 100%. Zero denominator yields N/A. Suppress trend points and changes when either relevant denominator is below 5 or absent; disclose uneven/incomplete coverage. Changes describe samples only, not population trends.

## E. Dashboard

First slice adds only a monthly sentiment trend chart, a monthly coverage/count/denominator table and explicit research period. Charts consume Pandas tables directly. Show incomplete, partial-month and sparse sampling warnings. The two generic sampling/uneven-coverage yellow banners are hidden at the user’s request; sampling methodology remains documented here and sample-bias captions remain visible. Monthly column headings have help descriptions. Retain TB1 aggregate charts, findings and Evidence Explorer without extra model calls on filtering. Prefer the existing chart renderer; use a small Plotly chart only if needed, managed via uv. No elaborate new dashboard features.

## F. Scope Boundaries

Reuse Streamlit, research.py and the existing classifier. No new sources, automatic repository discovery, multi-repository comparisons, background workers, Redis/PostgreSQL, FastAPI/Next.js, Docker Compose, LangGraph or orchestration infrastructure, significance testing, long-term raw storage, Deep/adaptive retrieval, or saved reports. Do not implement later PRD capabilities merely because they are documented.

## G. Acceptance Criteria

1. Default six-month input creates six correct, clipped calendar-month windows; tests cover year/leap-day boundaries and custom partial ranges.
2. Per-month allocations, 50 unique-issue global cap, request caps and five-call classification cap hold; bounded failures cannot trigger extra loops.
3. UTC creation timestamps assign issues to the right month independently of retrieval time.
4. Duplicate IDs are never classified or counted twice; PRs, invalid URLs and out-of-window records are rejected.
5. Synthetic fixtures through the same UI-called production pipeline verify exact monthly sentiment counts, denominators and percentages (including 2/5 = 40%).
6. Preserve TB1 multi-label aggregate complaint tests; monthly complaint comparisons are deferred.
7. Dashboard separates reported matches, retrieved N, relevant/classified N and the sentiment denominator, including incomplete/error metadata.
8. Empty/unclassified months show insufficient data; N < 5 is visibly sparse and suppressed on percentage trends; gaps are not interpolated.
9. Charts and findings equal the Pandas tables; filters never rerun classification.
10. Existing TB1 tests and UI remain working; tests/lint pass. Attempt credentialed live demonstration when available and report actual coverage honestly.

## H. Tracer Bullet Implementation Approach

Coding approval was provided with completed-month defaults, ascending/descending halves, and rate-aware sequential retrieval:

1. Build the smallest end-to-end slice: monthly GitHub retrieval → existing classifier in bounded batches → monthly Pandas aggregation → one sentiment trend chart with monthly N/coverage. Test the production path with synthetic fixtures, then attempt a live demonstration.
2. Review that working slice before adding monthly complaint trends and percentage-point comparisons.
3. Add within-month sampling improvements only if needed and within fixed budgets; rerun focused/regression tests and demonstrate limitations.

The first slice passed verification review. Monthly complaint trends and TB3 require separate explicit approval; stop after checkpointing this slice.

## I. First-slice validation checkpoint — 2026-10-08

Implemented on `feature/tb2-historical-sampling`; checkpoint commit authorized after verification. No merge or push authorized.
`uv run pytest -q`: 42 passed; `uv run ruff check .`: passed.
Live GitHub/OpenAI production pipeline: April–September 2026, 12 successful
search requests, 50 accepted issues, 50 classified, 46 relevant, 4 irrelevant,
no unknown labels or incomplete searches. Monthly retrieved Ns: 9,9,8,8,8,8;
sentiment denominators: 9,9,7,8,5,8. GitHub reported 1,112 matches across the
disjoint windows. These are observed exploratory-sample aggregates, not a
population trend or a semantic accuracy benchmark. Fresh classification used
the current prompt; manual classification accuracy review remains outstanding.

Live aggregate results rendered via Streamlit AppTest; monthly table equality
was checked. Local server health returned 200. The subsequent browser verification below completed visual inspection of the
finished live chart. No raw live source text or individual
classifications were saved. Monthly complaint trends remain deferred. Stop for
human review before further changes.

### Browser verification follow-up — 2026-10-08

Opened the actual local dashboard at `http://127.0.0.1:8512` and submitted
the default historical form through its Research button. The live run displayed
50 retrieved/classified issues from 1,112 reported matches, 44 relevant and
6 excluded. Visually inspected the rendered historical chart and monthly
coverage table. Negative filtering showed 26/50 rows; adding reliability
showed 18/50. Findings remained unchanged. Expanded the source inspector and
verified its canonical GitHub destination. Switching to Recent (TB1) restored
the recent 30-day form defaults. Browser console error check returned none.

A sentiment spot check prompted discussion about whether constructive ecosystem
contributions count as positive. The user considers that interpretation reasonable;
no classifier prompt change was made. Classification quality remains unbenchmarked
and labels can vary across runs; dashboard verification does not establish semantic accuracy. No raw issue content or item-level labels were saved.
The browser-testing requirement is now recorded in AGENTS.md.

## J. Final verification and limitations

Human verification approved on 2026-10-08. The final checkpoint reruns pytest
and Ruff; see TB2_HANDOFF.md for the results. An additional verification-only
synthetic check supplied oversized responses and inspected all outgoing queries:
50 accepted issues, 12 requests, five classification batches of ten.

Known limits: deterministic edge-biased sampling, sparse/uneven relevance,
unallocated capacity after failures, model interpretation/truncation and label
variability. GitHub ties at identical timestamps are not guaranteed. Retrieval
checks its 180-second budget before calls/waits; HTTP timeouts apply per network
phase, so this is not a hard whole-run wall-clock guarantee. The same synchronous
process performs classification afterward. Column help is configured for every
monthly column, but automation did not confirm the actual hover popup.

No raw source text, individual classifications, credentials or report data are
part of the checkpoint. Monthly complaint trends, Quick/Deep, adaptive retrieval
and saved reports remain deferred.
