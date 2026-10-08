# Community Research Analyst — MVP 1.0 Product Requirements Document

**Status:** Approved product direction; lightweight prototype scope proposed for implementation  
**Date:** 2026-10-07  
**Audience:** Implementing engineer / Codex  
**Companion:** [`TECHNICAL_DESIGN_CODEX.md`](TECHNICAL_DESIGN_CODEX.md)

## 1. Product summary

Build a **single-user, local-first research dashboard** that answers natural-language questions about developer experiences in technical communities. The engine retrieves genuine source discussions, asks an LLM to interpret and classify them, computes all numerical metrics in Python, and produces charts plus traceable links to evidence.

**MVP 1.0 is deliberately smaller than the long-term PRD:** one Streamlit process; GitHub Issues as the first live source; synchronous research; no authentication or infrastructure services. This is intended to establish that the research **analysis** is useful before investing in the final web platform.

Example supported question:  
> How has sentiment about LangChain changed across GitHub Issues in the last six months, and what complaints appear most often?

**Evidence caveat:** Because GitHub Issues primarily capture bugs, questions, and feature requests, sentiment measured here describes **sampled GitHub Issues**, not all developers or overall market sentiment.

## 2. Target users and purpose

The eventual intended audience combines:

1. **DevRel and product teams** investigating complaints, user experience, and product gaps.
2. **Developers and engineering leaders** researching practical difficulties, framework comparisons, and technology choices.

MVP 1.0 is tested by **one local operator**, without sign-in or collaboration.

## 3. Non-negotiable product principles

- **No invented data:** Counts, percentages, trends, and chart values must come from executable, testable Python aggregation.
- **No invented citations:** Every cited URL/ID must be present in retrieved source records.
- **Classification is separate from computation:** LLM assigns meaning; Pandas computes numbers.
- **Visible coverage:** Each report shows the sampled repository, time range, sample size, sampling limitations, and excluded/unclassified records.
- **Synchronous simplicity:** One-process execution is acceptable even if browser disconnection or refresh interrupts a run.
- **Constrained autonomy:** The model may propose search/refinement ideas, but Python enforces the source, repository, item, round, request, and model-call limits.
- **Raw evidence is session-only:** Persist only permitted report summaries, aggregates, methodology, and canonical source references—not GitHub issue bodies or item-level classification records.

## 4. MVP 1.0 — In scope (P0)

### FR-01: Streamlit input and local operation

- Streamlit app served on localhost.
- Enter a natural-language research question.
- Choose **Quick** or **Deep** mode.
- Select or supply a public GitHub repository in `owner/repo` form; include a convenient default for the LangChain demo.
- Show actionable error messages for missing configuration, no results, API failures, and model errors.

### FR-02: Live GitHub Issues retrieval

- Retrieve public GitHub **issues**, excluding pull requests, through GitHub REST search.
- Default research window: last six months for the demonstration; the UI can offer other supported time windows.
- Fetch capped issue titles, bodies, created-at timestamps, canonical URLs, IDs, and repo metadata.
- Prefer month-stratified collection to avoid confusing a sample of recent issues with a six-month time trend.
- Track total matches (as reported by GitHub), retrieved counts, API incompleteness, and collection failures.
- **Not in v1.0:** comments, GitHub Discussions, Reddit, Hacker News.

### FR-03: Lightweight research planning

- Parse a question into a structured plan: topic/entity, repository, time window, suggested GitHub search terms, desired metrics.
- Validate repository and dates in Python; reject unknown repository mappings rather than hallucinating them.
- GitHub searches must be constrained by an allowed repository and a time window.
- For Deep mode, optionally refine retrieval once if gaps are detected. No general-purpose autonomous tool execution.

### FR-04: LLM qualitative classification

For each relevant issue, classify:

- `relevance`: relevant / irrelevant / uncertain.
- `sentiment`: positive / negative / neutral / mixed / unclear.
- `complaint_themes`: zero or more from debugging, documentation, API stability, reliability, performance, integrations, usability, other.
- `experience_type`: firsthand-reported / secondhand / hypothetical / unclear.

One issue is one observation. Sentiment is entity-specific and a bug report **does not automatically mean negative sentiment**.

Use typed structured outputs and schema validation. Mark unsuccessful classifications as unclassified; do not invent labels.

### FR-05: Deterministic analysis

- Count collected, relevant, excluded, classified, and unclassified issues.
- Calculate sentiment distribution with explicit denominator.
- Calculate complaint counts and prevalence with explicit denominator (multi-label themes may sum above 100%).
- Compute monthly sample trends and monthly `n`, suppressing confident interpretations for thin/uneven samples.
- Implement all arithmetic in Pandas/Python; no LLM-computed numerical results.

### FR-06: Research report and dashboard

Report contains:

1. Scope, time period, repository, source, and research mode.
2. Concise evidence-supported executive findings.
3. Sentiment distribution and monthly trend (when data permits).
4. Top complaint themes with counts and percentages.
5. Evidence explorer with clickable, **actual GitHub issue URLs**.
6. Methodology and warnings about selection bias, incomplete searches, or low sample size.

Chart and KPI values come from `analytics.py`. Citation references in findings must resolve to the captured manifest. Use Streamlit plus Plotly/Pandas components.

### FR-07: Quick and Deep research profiles

| Setting | Quick | Deep |
|---|---:|---:|
| Maximum total candidate issues | 50 | 150 (up to 120 baseline + 30 targeted) |
| Maximum retrieval rounds | 1 | 2 |
| Intended purpose | Fast directional read | Wider/deeper sample and optional refinement |
| LLM classification | Yes | Yes |
| Evidence-backed report | Yes | Yes |

These are **starting configuration values**, not empirical performance or latency guarantees. Limit network retries, API calls, LLM calls, and request timeouts in both modes. Stop gracefully on budget exhaustion.

### FR-08: Session-only evidence + saved report

- Hold raw issue content and classifications in Python memory / active Streamlit session; do not write them to report JSON, logs, or a persistent cache.
- Provide a **Clear current research** action that discards in-memory raw data and classifications.
- Save a sanitized JSON report under local `data/reports/`, including permitted aggregate metrics, narrative, source URLs, coverage metadata, and report timestamp.
- On later app runs, list/reopen saved sanitized reports even if the original evidence is no longer available.
- Report should clearly indicate that the original raw evidence is no longer cached.
- Full automated session expiration/TTL enforcement is deferred; initial Streamlit sessions are ephemeral and clearing data must work explicitly. Do not claim guaranteed secure erasure of memory after browser disconnect.

### FR-09: Testable vertical slice

The app must work with deterministic, **synthetic** fixture data without live API keys for test execution, and with real GitHub data and an LLM key for live demonstration.

## 5. MVP 1.0 — Explicitly out of scope

- Next.js / FastAPI / microservices.
- Redis, Celery, MinIO, PostgreSQL, Docker Compose.
- Background workers, durable job queues, progress replay, and robust cancellation.
- Arbitrary model-generated Python code execution.
- GitHub Discussions and comments, Reddit, Hacker News.
- Full cross-source or cross-framework benchmarking.
- Persistent raw discussion indexes and vector databases.
- Robust cross-refresh contextual follow-up research.
- Multiuser accounts, auth, organizations, billing, or sharing.
- Scheduled monitoring and alerts.
- Statistical population-level inference or calibrated confidence scores.

## 6. MVP 1.0 user journey

1. Open local Streamlit app.
2. Enter a question and choose `Quick` or `Deep`.
3. Choose a repository (default `langchain-ai/langchain`).
4. Click **Research**; progress status displays truthful completed steps.
5. App runs a bounded plan → GitHub search → classify → compute → synthesize/validate pipeline synchronously.
6. Dashboard displays chart-backed findings, sample denominators, warnings, and original links.
7. User may filter visible results without calling the LLM again.
8. User may save/reopen the sanitized report or clear in-memory evidence.

## 7. Quality criteria and acceptance tests

**AC-01:** One `uv` command starts the Streamlit app locally.

**AC-02:** A real GitHub query returns authentic issue IDs, dates, and canonical `github.com` URLs with `is:issue` exclusion of PRs.

**AC-03:** On synthetic fixtures, the classifier input and output obey the typed schema; invalid outputs are handled gracefully.

**AC-04:** Exact numeric tests verify sentiment denominators, multi-label complaint prevalence, monthly counts, and empty-input handling.

**AC-05:** Every source linked in the findings or evidence explorer is present in the retrieval manifest. No fabricated citations are allowed.

**AC-06:** A report has scope, numbers, at least one relevant chart when data permits, original sources, and limitations.

**AC-07:** Quick and Deep have different resource limits; exceeding limits does not cause unbounded execution.

**AC-08:** Saving a report writes no raw issue bodies, comment bodies, or item-level classifications to disk.

**AC-09:** Clearing current research removes source-text objects from active Streamlit state while sanitized saved reports remain usable.

**AC-10:** Offline tests require neither `OPENAI_API_KEY` nor `GITHUB_TOKEN`.

**AC-11:** Missing credentials, rate limits, empty results, malformed model outputs, or incomplete GitHub search coverage produce visible errors/warnings rather than invented findings.

## 8. MVP 1.0 demonstration success

A single operator can ask the LangChain/GitHub question, see genuinely retrieved issue records, obtain a meaningful complaint breakdown and trend/coverage chart with computed denominators, inspect real citations, and reopen the sanitized saved report. The interface clearly says the evidence is an **exploratory sample of GitHub Issues**.

Demo outcome is **research usefulness and evidence traceability**, not a guaranteed 30-second runtime or statistical representation of the developer population.

## 9. Versioned roadmap: what comes later

| Target | Planned capabilities | Why deferred |
|---|---|---|
| **1.1 — Contextual investigation** | Follow-up question using active-session evidence, Quick→Deep expansion, immutable linked report versions, more robust retention and explicit idle expiry | Stateful orchestration beyond the first demo |
| **1.2 — Broader GitHub & comparison** | GitHub issue comments, GitHub Discussions, multiple repo/framework comparisons, better entity matching and source-aware normalization | Expand evidence quality after the pipeline works |
| **1.3 — Additional communities** | Hacker News discovery + original threads; Reddit only with approved access, usage, and retention terms; per-source coverage disclosures | External source/API limitations and compliance |
| **1.4 — Reliability & evaluation** | Human-labeled benchmark, better classification calibration, regression dataset, stronger contradictory-evidence checks, improved cost/latency telemetry | Requires empirical data |
| **2.0 — Production web application** | Next.js + FastAPI, durable background jobs, Redis queue, PostgreSQL, temporary object storage as justified, robust cancellation, report progress/replay, authenticated accounts and collaboration if demanded | Operational complexity unjustified for prototype |
| **Future** | Continuous monitoring, alerts, scheduled reports, team workspaces, export, enterprise controls | Outside on-demand MVP concept |

Later phases do **not** automatically authorize storing Reddit/other third-party content. Source-specific policies govern access, model processing, excerpts, and retention.

## 10. Resolved and remaining decisions

**Confirmed:** single-user local-first, Streamlit/Python, uv, synchronous research, GitHub Issues first, LLM classification + deterministic metrics, Quick/Deep, no infrastructure stack, temporary evidence, permitted saved aggregate reports.

**Implementation defaults (not previously approved as hard product commitments):** Python 3.12, LangChain example repo `langchain-ai/langchain`, Quick=50 items/1 round, Deep=150 items/2 rounds, initial six-month demo window, local report JSON storage, initial report-only reopened state.

**Still open and nonblocking for demo:** preferred model and per-run dollar budget, final active-session idle timeout, benchmark acceptance threshold for qualitative labels, final production access strategy for Reddit, and eventual deployment architecture.

## 11. Source and library documentation

- [uv project management](https://docs.astral.sh/uv/guides/projects/)
- [Streamlit Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state)
- [GitHub REST search](https://docs.github.com/en/rest/search/search)
- [OpenAI Python structured output helpers](https://github.com/openai/openai-python/blob/main/helpers.md)
