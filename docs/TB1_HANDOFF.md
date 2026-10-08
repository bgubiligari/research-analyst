# TB1 Handoff

## A. TB1 Objective

Validate the real end-to-end path: GitHub Issues → OpenAI classification → Pandas analytics → Streamlit dashboard. TB1 is implemented; TB2 specification preparation only is approved. No TB2 coding is authorized.

## B. Implemented Functionality

The form accepts a question, public `owner/repo`, and date range (default recent 30 days). One bounded GitHub search retrieves up to 10 newest issues, excluding PRs and rejecting invalid dates, URLs and duplicate IDs. OpenAI typed output provides relevance, sentiment and a list of complaint themes. Invalid/missing classifications are disclosed.

Pandas calculates sentiment and multi-label complaint counts and percentages. Deterministic Research Findings, coverage cards, sentiment bars, and compact complaint bars (0–100%, direct percentage labels) use these aggregates. Evidence Explorer provides original issue links, titles, dates, labels, in-memory filters and a description inspector. Coverage and sampling caveats are visible. Findings are templates, not model-written narrative.

## C. Architecture and Key Files

Local-first, synchronous Streamlit + Python, managed with `pyproject.toml` and `uv.lock`; no databases, workers or orchestration frameworks.

- `app.py`: form, truthful progress, session state, charts, tables and source inspector.
- `research.py`: schemas, GitHub retrieval, OpenAI boundary, `run_research`, Pandas metrics and presentation helpers.
- `tests/test_research.py`: synthetic production-pipeline, boundary, arithmetic and Streamlit tests.
- `tests/test_classification_policy.py`: synthetic schema/prompt contracts; not an accuracy benchmark.
- `.env.example` / `README.md`: configuration and commands. Local `.env` is ignored; use `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `GITHUB_TOKEN`.

## D. Important Design Decisions

GitHub Issues are the only source; one issue is one observation. Themes are independent list labels, counted once per issue per theme. A bug alone does not imply a debugging complaint. LLM interpretation is separate from Python arithmetic. Sentiment denominator is relevant classified issues excluding unclear sentiment; complaint denominator includes all relevant classified issues, including empty themes and unclear sentiment. Theme percentages may exceed 100% in total.

Links must match retrieved records. Raw evidence and classifications stay in active memory; no saved reports or persistent cache. Titles/bodies are capped at 500/5,500 characters. GitHub samples are not representative of overall developer sentiment.

## E. Test Status

On 2026-10-08: `uv run pytest -q` **32 passed**; `uv run ruff check .` **passed**; staged whitespace checks passed. Fixtures exercise the same `run_research` path as the UI and mock external boundaries.

Earlier credentialed live GitHub/OpenAI/UI smoke testing was performed in this conversation. The final chart was visually checked against the existing live session: 60%, 50%, 20% matched computed findings. No fresh live classification was performed for this documentation checkpoint. The latest positive-sentiment prompt clarification has not been verified with a fresh model call. Classification accuracy remains unproven; manual review found unsupported positive interpretation. Existing session labels were not overwritten. No raw source or model output is included in this handoff.

## F. Known Limitations

Newest-first, non-random sample of at most 10; the question guides classification but does not filter retrieval. No meaningful historical trends, multi-source research, durable background execution, Quick/Deep modes or report persistence. Refresh/disconnect can lose session evidence. Model labels can be wrong and input truncation can omit context; fixture success does not establish semantic accuracy.

## G. TB1 Git Checkpoint

Implementation commit: `1c890a2b45a5e9d843eb06ca0a94416272e48e1f` — `feat: complete TB1 GitHub research and classification dashboard` (builds on `f3b0f34`). Next: review `TB2_SPEC.md`, propose a minimal plan, and obtain explicit coding approval.
