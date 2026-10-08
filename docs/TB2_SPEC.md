# TB2 Specification — Historical Sampling and Sentiment Trends

**Status:** Specification preparation approved; coding not approved. This document proposes TB2 only. Read AGENTS.md and TB1_HANDOFF.md before proposing an implementation plan.

## A. Goal

Support questions such as “How have developer complaints about LangChain changed over the past six months?” Extend the working pipeline with calendar-month coverage rather than a single newest-first sample. Results describe sampled GitHub Issues, not a statistically representative population.

## B. Historical Sampling

- Split the selected date range into non-overlapping UTC calendar-month windows, clipping first/last windows to requested dates. Default: current month to date plus the preceding five calendar months (six buckets); disclose partial months.
- Configurable target of 10–20 issues per month (default 10), constrained by the existing **50-issue total TB2 budget**. Targets are not guarantees. Allocate capacity evenly across windows before retrieval; for six windows, 8 each plus one extra for the first two chronologically. Never silently increase the global cap to meet monthly targets.
- Baseline queries retain repository, public issue and creation-date constraints; exclude PRs. Do not add negative keywords or question-aware retrieval in TB2.
- First slice: one bounded request per month with explicit newest-first-within-month disclosure. Later within TB2, where practical, split months into early/late windows with fixed subquotas to spread coverage. No claim of randomness; no unbounded refill loop.
- Deduplicate globally using canonical repository/issue identifiers and validate original URLs. Preserve issue `created_at` separately from run retrieval timestamp; bucket only by creation time.
- Report GitHub match counts separately from actual accepted sampled counts. Unknown totals are N/A, never zero. Retain incomplete flags, errors, discarded counts, quota limits and strategy per month.
- Proposed bounds: at most two requests per calendar month, no automatic retries, no pagination beyond the allocated request/item budgets; stop or return explicitly partial coverage on rate limits. Retain HTTP timeouts. Any pagination enhancement must honor these caps and GitHub search limits.
- Classify accepted issues in batches of at most 10 using the existing engine, at most five calls for 50 issues; failed batches stay unclassified. Keep everything synchronous and in memory.

## C. Monthly Sentiment Analytics

For every requested month, including empty/error months, compute reported matches where available, retrieved count, relevant classified count, positive/negative/neutral/mixed counts and percentages, unclear count, unclassified count, and sentiment denominator. Also expose irrelevant/uncertain exclusions.

Reuse TB1: sentiment denominator = relevant classified issues with sentiment other than unclear. No classifiable evidence means N/A/insufficient data, not 0% negative. For denominator below 5, retain counts and warn; suppress percentage trend points. Never connect a line across missing/suppressed months. All arithmetic is Python/Pandas.

## D. Monthly Complaint Analytics

Compute theme counts, prevalence, deterministic top categories, and adjacent-month prevalence differences in percentage points. Denominator = all relevant classified issues, including empty themes and unclear sentiment. Count an issue once per theme; allow overlapping categories and totals above 100%. Zero denominator yields N/A. Suppress trend points and changes when either relevant denominator is below 5 or absent; disclose uneven/incomplete coverage. Changes describe samples only, not population trends.

## E. Dashboard

Add a monthly sentiment trend chart, then a compact complaint-theme trend visualization, a monthly coverage/count/denominator table and explicit research period. Charts consume Pandas tables directly. Show incomplete, partial-month, sparse and uneven sampling warnings. Retain TB1 aggregate charts, findings and Evidence Explorer without extra model calls on filtering. Prefer the existing chart renderer; use a small Plotly chart only if needed, managed via uv. No elaborate new dashboard features.

## F. Scope Boundaries

Reuse Streamlit, research.py and the existing classifier. No new sources, automatic repository discovery, multi-repository comparisons, background workers, Redis/PostgreSQL, FastAPI/Next.js, Docker Compose, LangGraph or orchestration infrastructure, significance testing, long-term raw storage, Deep/adaptive retrieval, or saved reports. Do not implement later PRD capabilities merely because they are documented.

## G. Acceptance Criteria

1. Default six-month input creates six correct, clipped calendar-month windows; tests cover year/leap-day boundaries and custom partial ranges.
2. Per-month allocations, 50 unique-issue global cap, request caps and five-call classification cap hold; bounded failures cannot trigger extra loops.
3. UTC creation timestamps assign issues to the right month independently of retrieval time.
4. Duplicate IDs are never classified or counted twice; PRs, invalid URLs and out-of-window records are rejected.
5. Synthetic fixtures through the same UI-called production pipeline verify exact monthly sentiment counts, denominators and percentages (including 2/5 = 40%).
6. Multi-label complaint fixtures verify overlap, per-theme deduplication, empty-theme inclusion and percentage-point changes.
7. Dashboard separates reported matches, retrieved N, relevant/classified N and both denominators, including incomplete/error metadata.
8. Empty/unclassified months show insufficient data; N < 5 is visibly sparse and suppressed on percentage trends; gaps are not interpolated.
9. Charts and findings equal the Pandas tables; filters never rerun classification.
10. Existing TB1 tests and UI remain working; tests/lint pass. Attempt credentialed live demonstration when available and report actual coverage honestly.

## H. Tracer Bullet Implementation Approach

After explicit coding approval and review of a minimal plan:

1. Build the smallest end-to-end slice: monthly GitHub retrieval → existing classifier in bounded batches → monthly Pandas aggregation → one sentiment trend chart with monthly N/coverage. Test the production path with synthetic fixtures, then attempt a live demonstration.
2. Review that working slice before adding monthly complaint trends and percentage-point comparisons.
3. Add within-month sampling improvements only if needed and within fixed budgets; rerun focused/regression tests and demonstrate limitations.

Stop at the TB2 gate for human review. TB3 is not approved. Proposed defaults (six bucket convention, quota allocation and request caps) are reviewable with the implementation plan; no application code is changed by this specification.
