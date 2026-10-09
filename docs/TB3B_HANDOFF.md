# TB3-B handoff — Quick and Deep Research

Completed and verification approved on `feature/tb3-adaptive-research`, based on
TB3-A checkpoint `aa1dfcb8dc0c921d2ba7a3b32db98489bc71c2da`.

## Workflow and budgets

Sampling (Recent / Historical) and research depth (Quick / Deep) are independent.
Quick is the default and performs baseline-only research, without an evidence-gap
evaluation. Deep reuses TB3-A's structured evaluator: sufficient evidence skips
retrieval; a valid permitted-theme gap allows one Python-constrained GitHub search.
Invalid decisions, retrieval failures and exhausted budgets preserve baseline
results with a visible explanation. Only clicking Research executes the pipeline.

| Configuration | Baseline issues | Additional candidates | GitHub calls | Model calls |
|---|---:|---:|---:|---:|
| Recent Quick | 10 | 0 | 1 | 1 |
| Recent Deep | 10 | 10 | 2 | 3 |
| Historical Quick | 50 | 0 | 12 | 5 |
| Historical Deep | 50 | 10 | 13 | 7 |

These are maximums. `research.py` enforces mode-specific limits before calls,
using the existing shared rate-limit state, 180-second retrieval budget and
best-effort total deadlines (Quick 360 seconds; Deep 450 seconds). Classification
batches contain at most ten issues. No pagination, quota refill or retry framework
is added. Repository, public-issue, date and source-identity validation remain.

Baseline and targeted evidence stay separate. Only new targeted issues are
classified; baseline duplicates reuse classifications. Supporting citations map
to accepted retrieved records. Targeted evidence never changes baseline counts,
sentiment, complaint prevalence, monthly coverage or historical chart datasets.
Raw sources and item-level labels remain in temporary session memory.

## Dashboard and result state

`app.py` stores the executed sampling/depth with each result. Changing either
selector without submission leaves the report's executed mode, charts and evidence
intact. Additional Investigation appears only for an executed Deep result and
shows its decision, reason, retrieval outcome and separate supporting links.
Evidence Explorer defaults to Relevant; All, Irrelevant and Uncertain intersect
with sentiment/theme filters. All includes unclassified records. Filters affect
only the in-memory table and make no GitHub/OpenAI calls.

## Verification

Final checkpoint commands: `uv run pytest -q` (**82 passed**),
`uv run ruff check .` (**All checks passed**) and `git diff --check` (**passed**).
Synthetic fixtures exercise the same production `run_research` path for all four
configurations, both Deep decisions, budgets, duplicate reuse, accepted citation
mapping and exact baseline metric/chart equality. Streamlit AppTest verifies
executed-mode preservation and filtering without another pipeline call. Existing
TB1, TB2 and TB3-A regressions pass.

Real GitHub/OpenAI browser checks on October 8, 2026 demonstrated all four modes;
both Deep modes naturally skipped retrieval when evidence was sufficient. After
the relevance refinement, a fresh Historical Deep run naturally investigated
performance: one targeted request, ten accepted candidates, one baseline duplicate,
three new supporting issues and seven total model calls. Relevant/All/Irrelevant/
Uncertain displayed 16/50/34/0 baseline rows without errors. Changing controls to
Recent Quick without Research retained Historical Deep, charts and supporting
evidence; counters stayed unchanged through filtering. The other three modes were
not rerun live after the filter-only refinement; current fixtures cover all four.
No fresh live classification was performed during this documentation checkpoint.

## Limitations and deferred work

Fixed theme keywords, deterministic sampling bias, truncated model inputs and
model variability remain; fixtures do not establish semantic classification
accuracy. Synchronous deadlines are best effort, and session disconnection may
lose active evidence. Deep does not guarantee additional retrieval. No known
remaining defect was found in verification. Monthly complaint trends, narrative
generation, saved reports, new sources and TB4 remain deferred and unapproved.
Preserve the lightweight Streamlit/Python/uv architecture and stop for human
review before any further implementation.
