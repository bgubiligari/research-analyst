# TB3-A Handoff — Bounded Adaptive Investigation

Implementation and final verification approved on 2026-10-08. Checkpoint branch:
`feature/tb3-adaptive-research`. TB3-B planning only is authorized; coding awaits
approval. No merge or push is authorized.

## Workflow and architecture

The Streamlit form optionally enables additional investigation through the same
synchronous `run_research` pipeline used by fixtures. `research.py` owns retrieval,
strict structured OpenAI classification/evaluation, budgets and Pandas analytics;
`app.py` renders results in session memory. No new dependencies or infrastructure.

After the existing Recent or Historical baseline, Python checks relevant evidence,
permitted themes and remaining request/model/time capacity. One structured LLM
evaluation sees at most six relevant examples, each capped at 200 title and 1,000
body characters, plus baseline counts and Python-approved themes. Decisions use a
strict boolean, known theme, finite reason code and validated example IDs. Extra
fields, incompatible decisions, invalid themes/IDs or evaluation failures skip
investigation with a visible explanation.

- **Skip:** sufficient baseline evidence produces no targeted search.
- **Investigate:** missing concrete detail or insufficient examples permits one
  Python-built search. Its repository, public-issue restriction, original dates,
  fixed theme keyword and descending creation order cannot be changed by the model.

At most ten targeted candidates are validated for issue type, identity, UTC date
and original GitHub URL. Duplicates against baseline retain existing labels and
are not reclassified. One batch classifies only new candidates. Only genuinely
relevant labels containing the selected theme appear as supporting links. Empty,
duplicate-only, invalid, incomplete or failed results preserve baseline findings.
The dashboard shows the decision, reason, theme, retrieval outcome, source links,
limitations and execution counters.

## Cohort separation and budgets

`ResearchResult.batch`, baseline labels and Pandas tables remain separate from
`ResearchResult.investigation` (targeted manifest, new labels, supporting issues).
Targeted evidence is qualitative; it never enters sentiment, complaint-prevalence,
monthly coverage or chart denominators.

| Bound | Historical baseline | Recent baseline |
|---|---:|---:|
| Issues | 50 | 10 |
| GitHub requests | 12 | 1 |
| Classification batches, at most ten each | 5 | 1 |

Investigation adds at most one request, ten candidates, one evaluator call and one
new-issue classification batch. Global enabled-run ceilings are 13 GitHub requests
and seven model calls; Recent's single baseline request/batch means it needs at
most two requests and three model calls. Failed calls consume capacity. There is
one targeted retrieval round, with no retries, pagination, refill or redistribution.

Requests are sequential and share rate-limit headers and a 180-second retrieval
deadline. Whole-run deadlines are 360 seconds without investigation and 450 with
it, checked before calls with bounded timeouts. Waiting that cannot fit stops
cleanly. Execution is synchronous and best effort, without background preemption.

Raw source text, item-level labels and results remain in active session memory.
Both OpenAI boundaries use `store=False`. No source cache, raw-evidence logging,
generated report files or credential persistence was added; `.env` stays ignored.

## Verification

Final checkpoint commands: `uv run pytest -q` — **68 passed**;
`uv run ruff check .` — **All checks passed**; `git diff --check` — passed.
All existing TB1/TB2 tests pass. Fixtures exercise the production orchestrator and
prove both decisions, strict validation, exact baseline table/chart equality,
item/call caps, deduplication, citation mapping, empty/failure outcomes, deadline
stops, shared pacing, targeted 403/429 handling and dashboard filtering without
new calls. Synthetic records are labeled fixtures, not live research.

Both paths were also verified in the real Streamlit browser using genuine
GitHub/OpenAI calls and the unchanged classifier/evaluator prompts:

- **Skip:** April–September 2026 reliability question; 50 baseline issues labeled,
  22 relevant, 12 searches, six model calls and zero targeted requests. The live
  evaluator judged evidence sufficient.
- **Investigate:** a natural question comparing reliability failures with slow
  streaming/response times selected performance for missing concrete detail.
  Twelve baseline searches plus one targeted search returned ten accepted
  candidates, including one baseline duplicate. Nine new candidates went to
  classification; three supporting issues were confirmed. Seven total model
  calls. All three supporting links opened as matching public GitHub Issue pages.

A temporary labeled synthetic browser harness independently displayed the valid
investigate outcome and counters; it was stopped and closed after verification.
A stale pre-existing Streamlit process was restarted before successful live checks.
No fresh live run was made during the final checkpoint; the approved live results
above are from preceding verification. No live source text or item-level labels
are included in this document.

## Known limitations and next step

Fixed theme keywords limit recall; newest-first targeted results are biased and
cannot estimate prevalence or fill missing months. Truncated examples can miss
context. Model decisions/labels vary and semantic accuracy is not independently
benchmarked. Slow baseline work or rate limits can leave no retrieval time for
investigation. Timeouts are best effort, not a hard whole-run cancellation.
Refresh/disconnect can lose session evidence; restart Streamlit when imported
Python code changes to avoid stale modules.

Next proposed slice: TB3-B Quick/Deep UI over this validated workflow. Keep Recent
and Historical sampling independent, preserve smaller approved budgets and cohort
separation, and obtain explicit coding approval first. No TB4 work is included.
