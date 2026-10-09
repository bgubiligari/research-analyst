# TB4-A handoff — Evidence-backed research narratives

Completed and final verification approved on `feature/tb4-research-reports`.
The prototype is frozen for the team demonstration. TB4-B is not authorized.

## Implemented workflow

The existing synchronous `research.py:run_research` pipeline retrieves GitHub
issues, classifies them, calculates Pandas metrics, optionally investigates a
Deep-mode evidence gap, then attempts one structured narrative generation.
`app.py` displays summary, findings, complaints and eligible historical
interpretation alongside the existing charts, coverage table and source explorer.
Controls and filters do not initiate API calls; the report preserves its executed
configuration. Narrative failure leaves deterministic findings and charts intact.
If a summary is rejected, accepted key findings may serve as the labeled fallback.

## Support and validation

Python selects at most six relevant baseline and two confirmed targeted examples.
Titles are bounded to 200 characters, excerpts to 1,200, and serialized model
context to 24,000 UTF-8 bytes. Stable evidence IDs include cohort prefixes;
Python resolves URLs from retrieved records. Baseline and targeted evidence stay
separate, and targeted records never affect baseline denominators or charts.

Aggregate claims display canonical Pandas metric references without individual
issue citations. Qualitative claims require supporting evidence IDs, relevant
records and permitted complaint themes. Combined findings retain separate metric
and issue provenance. Unknown IDs/metrics and unsupported prose are rejected per
claim without discarding valid siblings. Historical prose is Python-rendered:
adjacent eligible months require N >= 5; patterns require at least three consecutive
eligible months. Sparse months break segments and remain suppressed chart gaps.

GitHub Bug/Feature metadata is retained for narrative context and citation labels,
without changing classifier inputs. Instructions distinguish reported bugs,
documentation defects, feature requests and provider-specific integration gaps;
feature requests alone cannot establish confirmed production failures.

Rejected-claim diagnostics include bounded candidate text, claim index/section,
reason code, explanation and referenced IDs/keys. They appear only in a collapsed
developer expander and live with the active result in Streamlit session memory.
Replacing/clearing active evidence also removes its diagnostics. No report writer,
source cache, raw response logging or persistence was added.

## Resource limits

| Configuration | Baseline issues | Targeted candidates | GitHub requests | Total model calls |
|---|---:|---:|---:|---:|
| Recent Quick | 10 | 0 | 1 | 2 |
| Recent Deep | 10 | 10 | 2 | 4 |
| Historical Quick | 50 | 0 | 12 | 6 |
| Historical Deep | 50 | 10 | 13 | 8 |

Limits are maximums. The additional capacity is reserved for one narrative call;
classification/evaluation limits are unchanged. Narrative output is capped at
2,000 tokens, with no retries, `store=False`, a maximum 30-second request timeout
and the existing overall deadlines (Quick 360 seconds; Deep 450 seconds).
Sequential retrieval retains shared GitHub rate-limit pacing and deadline checks.

## Verification and limitations

Checkpoint checks: `uv run pytest -q` — **130 passed**;
`uv run ruff check .` — **All checks passed**; `git diff --check` — **clean**.
Synthetic tests exercise the production pipeline across all sampling/depth paths,
baseline invariance, provenance separation, specific rejection reasons, fallback,
session clearing, budgets and no diagnostic/source writes or logs.

The final live Historical Quick browser check on October 9, 2026 used configured
model `gpt-4.1-mini`: 50 issues retrieved/classified, 39 relevant, 12 searches and
6 model calls. Narrative usage was **5,565 input + 786 output = 6,351 tokens**.
Three rejected claims were inspectable while accepted findings remained visible.
The chart and coverage table rendered correctly; April N=4 was suppressed and
historical interpretation covered May–September without issue citations.
Accepted bug and feature-request links were checked against original descriptions.
An earlier verification run exposed shortened evidence IDs; the final context and
instructions remove that ambiguity. Other live narrative mode combinations were
not rerun during the final correctness checkpoint; they have offline coverage.

Filtering is intentionally frozen and conservative: harmless wording can be
rejected. Reference validation does not prove semantic entailment; truncated
excerpts, classifications and model prose can omit scope or be inaccurate.
GitHub Issues use deterministic, non-random sampling with uneven coverage and
describe sampled issues rather than all developers. The resolved provider model
version is not recorded. Sessions are ephemeral; persistence, saved reopening and
a dedicated Clear action remain deferred to separately approved TB4-B work.
