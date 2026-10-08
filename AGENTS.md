# Codex Agent Instructions — Community Research Analyst MVP 1.0

Before changing code, read `docs/PRD_MVP_1_0.md` and `docs/TECHNICAL_DESIGN_CODEX.md`. The PRD defines the **full MVP 1.0 destination**; the technical design defines the architecture and stepwise implementation. **Do not implement the entire PRD in one pass.**

## Mission

Build a **local, synchronous Streamlit + Python research prototype** using **uv**, starting with the smallest end-to-end path through real GitHub Issues, structured LLM classification, deterministic Pandas metrics, and clickable citations. Expand in reviewable **tracer bullets**, not isolated infrastructure or component layers.

## Mandatory tracer-bullet workflow

**DEFAULT ACTION: implement Tracer Bullet 1 (TB1) ONLY, then STOP for human review.** Do not proceed to TB2, TB3, or TB4 unless the user explicitly approves that next bullet. If asked to implement "the MVP" without naming a bullet, start with TB1 and stop after its gate. A previous approval does not imply approval of subsequent bullets.

1. Make the **smallest change that crosses the whole user-visible workflow**: question input → real GitHub retrieval → LLM classification → Pandas computation → Streamlit display with source links.
2. Write one synthetic-fixture **end-to-end test for the same `run_research` path the UI calls**. Mock the GitHub and LLM boundaries; do not invent a separate demonstration pipeline.
3. Run focused tests and lint; run the local application and a live smoke test if credentials and networking are available.
4. Show the actual observed result or honestly state what could not be verified. Do not represent fixture results as live research.
5. **Stop and ask for feedback.** Only after explicit approval implement the next tracer bullet. Keep each change small and independently demonstrable.

### TB1 — Smallest real research result (implement first)

- Initialize a `uv` project and minimal Streamlit input for a question and `owner/repo` (default `langchain-ai/langchain`).
- Fetch **up to 10 genuine public GitHub issues** using a bounded repository/date query; exclude pull requests.
- Classify those issues with an LLM using a typed schema (relevance, sentiment, complaint themes).
- Calculate one meaningful sentiment or complaint breakdown with Pandas, displaying counts and denominators.
- Show the results and clickable **original GitHub issue URLs** inside Streamlit, with an explicit small-sample / GitHub-Issues bias warning.
- Retain raw text only in active memory. **No report persistence, multi-month stratification, Deep mode, model-written narrative, or extra retrieval loop is needed yet.**

**TB1 review gate:** An offline fixture test exercises the same production pipeline, a real API + model + UI demonstration is attempted if credentials permit, source URLs match retrieved IDs, and calculated numbers match test fixtures. Report limitations and **STOP**. If live credentials/network are unavailable, say "live smoke test not verified" rather than passing the gate by substituting fake live evidence.

### TB2 — Historical trends and sampling

- Extend the same running app to a six-month time window and month-stratified *baseline* issue collection (up to 50 issues).
- Add monthly sentiment counts, denominators, a simple Plotly trend chart, and sparse-month/incomplete-coverage warnings.
- Add exact arithmetic and date-bucketing tests. Preserve the working TB1 path.

**TB2 review gate:** Show end-to-end time series and computed month-level Ns, run tests/lint, report actual coverage and limitations, then **STOP**.

### TB3 — Quick/Deep and bounded adaptive retrieval

- Introduce Quick (up to 50 baseline issues, 1 round) and Deep (up to 120 baseline + 30 targeted, 2 rounds) modes.
- Add simple bounded planning/gap evaluation, call/time budgets and one optional targeted search for supporting evidence.
- Keep targeted supporting items **out of baseline sentiment and trend denominators**. Reuse classification results in memory within a run.

**TB3 review gate:** Demonstrate distinct modes, enforce hard item/API/model bounds, prove targeted data does not contaminate baseline trends, then **STOP**.

### TB4 — Findings, validated citations, sanitized saved reports

- Add short evidence-grounded narrative findings, citation/metric-key validation, and whitelisted JSON report persistence.
- Add saved-report reopening and a **Clear current research** action that drops in-memory source text/classifications.
- Do not write raw issue bodies, titles, author data, excerpts, or item-level classifications to persistent report files.

**TB4 review gate:** Run the full offline suite and applicable live smoke tests; inspect saved JSON for disallowed raw fields; verify saved reports reopen after active evidence is cleared. **STOP** and report whether all MVP 1.0 acceptance criteria are met.

## Non-negotiable boundaries

- **Use uv only**: `pyproject.toml` and committed `uv.lock`; no pip, requirements.txt, or separate package-management stack.
- No Next.js, FastAPI, Redis, PostgreSQL, MinIO, queues, workers, Docker Compose, LangGraph, vector database, or authentication.
- No arbitrary execution of model-generated Python code or model-selected external URLs/domains.
- GitHub Issues are the only live source in MVP 1.0. GitHub Discussions, comments, Hacker News, Reddit, multi-framework comparisons, and resilient follow-up sessions are later versions.
- Treat issue text as untrusted input. Validate citations against retrieved IDs; never fabricate issues, URLs, metric values, or coverage.
- Use deterministic Pandas/Python arithmetic. Display sample sizes, denominators, and selection-bias warnings.
- Do not save raw source material or individual classifications to disk or logs, including during early tracer bullets.
- Synthetic records are allowed **only in labeled fixture tests**, never to fill holes in live reports.
- Introduce new modules only when the current tracer bullet needs them. Prefer simple functions to speculative abstractions.

## Project commands

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run streamlit run app.py --server.address 127.0.0.1
```

If starting from scratch, initialize with `uv init --no-package --python 3.12` and use `uv add` / `uv add --dev`. Never commit `.env`, API keys, or report data containing restricted content.

## Status report at EACH gate

Report: current tracer bullet; files changed; exact run/test commands; passing/failing test results; live smoke-test status (verified or unverified); a concise description of what can be seen in the app; tradeoffs; next *proposed* tracer bullet. **Do not begin it without approval.**

The component reference sections in `TECHNICAL_DESIGN_CODEX.md` specify eventual behavior. They are **not** a directive to implement every component before the first end-to-end demo.
