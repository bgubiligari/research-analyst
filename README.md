# Community Research Analyst — TB1 through TB4-A

Local synchronous Streamlit research using public GitHub Issues, typed OpenAI
classification, and deterministic Pandas sentiment counts and historical trends.
Optional supporting investigation stays separate from baseline statistics.
Quick and Deep research depth are independent of Recent/Historical sampling.
No saved reports or synthetic live data.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and OPENAI_MODEL; GITHUB_TOKEN is optional.
uv run streamlit run app.py --server.address 127.0.0.1
uv run pytest -q
uv run ruff check .
```

Open http://127.0.0.1:8501. Enter a question, owner/repo and dates; click Research.
Historical sampling defaults to six completed UTC months (see below). In Recent
(TB1), the default window is 30 days. One newest-first, unfiltered repository/date search
returns at most 10 candidates; PRs and invalid records are discarded without
refilling. One structured model call (no retries, 30-second timeout) classifies
bounded issue text (500 title + 5,500 body characters). GitHub timeout is 20 seconds.

The sentiment denominator includes only relevant, successfully classified issues
with non-unclear sentiment. Counts, exclusions, coverage, original links and bias
warnings are visible. Missing credentials/model errors leave retrieved issues
unclassified with an explicit warning. Source failures display a safe error.

Complaint prevalence counts each theme once per relevant classified issue. Its
denominator includes issues with no complaint themes and unclear sentiment;
multiple themes can make percentages sum above 100%. Counts, denominators,
percentages and a prevalence bar chart reuse the existing classification batch.
The question guides classification only, not retrieval: narrow questions may
have little relevant evidence in this small repository-level sample.

The dashboard includes deterministic Research Findings, coverage cards, sorted
horizontal complaint bars, and an evidence explorer with issue links, titles,
sentiment, themes, relevance and creation dates. Sentiment/theme filters use
session data only and do not rerun research. Full metric tables and detailed
counts remain available in expanders; sampling caveats are notes at the bottom.
Synthetic classification cases verify mapping and arithmetic, not model accuracy;
review the labels against the original linked issues to assess classification quality.

The classifier defines each complaint theme independently: bugs alone do not imply
debugging or usability difficulties, and provider names alone do not imply an
integration complaint. Sentiment reflects substantive content, including adverse
functional defects, rather than issue templates or feature-request wording.
The Evidence Explorer's description expander reuses the retained body (up to
5,500 characters) and original link without fetching or classifying again.
The eight synthetic policy examples exercise schema, prompt wiring and arithmetic;
they do not assert deterministic live-model labels or constitute an accuracy benchmark.

Results exist only in active session memory; refresh/disconnect may lose them.
There is no source-text logging, persistent cache or report saving. `store=False`
is used for the model response; this does not promise zero provider-side retention.
GitHub issues overrepresent problems; this tiny non-random sample cannot describe
all developers. Historical charts describe the sampled issues only. Classification
is model interpretation.

Offline evaluation runs synthetic fixtures through the same `run_research`
function the UI calls; tests require no credentials and do not enable a fake-live
mode in the app. The first TB2 slice has passed human verification review.

## TB2 first slice (verification approved)

Historical sampling is the default; Recent (TB1) preserves the 10-issue workflow.
Historical dates default to the last six completed UTC calendar months. Custom
ranges support up to six calendar buckets, with partial-month warnings.

The 50-issue capacity is evenly allocated oldest-first (six months: 9, 9, 8, 8,
8, 8). Each clipped interval is split into disjoint date halves: early ascending
creation order, late descending; odd days and quotas go early. This deterministic
selection favors monthly edges and is not representative or random. GitHub does
not guarantee a tie-break for equal creation timestamps. No pagination, refill,
redistribution or automatic retries occur.

At most 12 sequential searches run within a 180-second retrieval budget. Rate
headers pace requests; waits are bounded and reported. Budget exhaustion or
403/429 preserves prior evidence and marks remaining intervals unattempted.
Without a GitHub token, search rate limits can cause partial coverage. At most
five batches of ten use the existing classifier; failures remain unclassified.

Monthly counts and coverage separate reported matches, returned/rejected records,
accepted issues and classifications. Unknown totals and zero-denominator
percentages are N/A. Sentiment N excludes irrelevant, uncertain and unclear
labels. N < 5 points are suppressed; chart lines never cross these gaps.
Monthly complaint trends remain deferred. All source text and labels stay in
memory; no new persistence or model-generated narrative is introduced.


## TB3-A — Optional additional investigation

Choose **Deep** research before running research. **Quick** is the default and
performs baseline-only research, with no gap evaluation or targeted search. Python checks relevant
baseline evidence, supported themes and remaining budgets. One strict structured
model decision can accept existing evidence or request one known complaint theme.
Python constructs the complete repository/public-issue/date query using a fixed
keyword mapping; model output cannot supply queries, URLs or budgets.

Historical baseline remains capped at 50 issues, 12 searches and five batches.
The optional path allows one extra search (10 candidates), one evidence-gap call
and one new-issue classification batch: at most 13 GitHub requests and seven model
calls. Recent mode retains its 10-issue baseline. No retries, pagination or refill.
Requests share header-aware rate-limit state and a 180-second retrieval deadline;
total execution deadlines are 360 seconds without investigation and 450 seconds
with it, checked before calls with bounded per-call timeouts.

Recent Quick is capped at 10 baseline issues, one GitHub request and one model
call; Recent Deep adds at most 10 candidates, for two requests and three model
calls. Historical Quick uses 50/12/5 and Historical Deep uses 50+10/13/7. Limits
are ceilings, not required usage. Changing sampling, depth or evidence filters
makes no API calls; only Research submits a run. Completed reports show the
executed configuration until replaced by another submitted run.

Duplicate issues retain their baseline classifications. Only newly retrieved,
validated, relevant issues labeled with the selected theme appear as supporting
links. Targeted evidence never enters baseline metrics, monthly coverage or charts.
The Additional Investigation section reports decisions, counts and limitations.
See [TB3-A handoff](docs/TB3A_HANDOFF.md) and [TB3-B handoff](docs/TB3B_HANDOFF.md)
for verification and limitations.

Evidence Explorer defaults to **Relevant** issues. Its relevance selector offers
Relevant, All, Irrelevant and Uncertain; All also includes unclassified issues.
Relevance, sentiment and complaint-theme filters intersect in memory and affect
only the evidence table. They never alter classifications, findings or charts.

## TB4-A — Evidence-backed narrative (completed and approved)

Research now permits one additional structured narrative call after the existing
pipeline. Python selects at most six relevant baseline excerpts and two confirmed
targeted supporting excerpts, assigns stable evidence IDs and resolves original
URLs. Each claim is validated independently; unknown references, unsupported
metrics and model-authored quantities are omitted. Python renders canonical
Pandas counts/denominators/percentages and historical comparisons. Two consecutive
months with N >= 5 permit descriptive comparisons; three or more permit cautious
sampled patterns. Suppressed months remain gaps. Narrative failure preserves
deterministic findings and charts with a visible unavailable message.

Current total model-call caps, including narrative: Recent Quick 2, Recent Deep 4,
Historical Quick 6, Historical Deep 8. Retrieval/classification/investigation
limits above remain unchanged. One narrative attempt uses at most 2,000 output
tokens, a bounded input manifest and the existing overall deadline. Aggregate
API token usage is displayed when available. Earlier TB3 call counts above describe
the workflow before adding narrative. Controls and filters never regenerate it.

Narrative validation checks references and arithmetic, not semantic entailment.
Review cited originals manually; excerpts, classifications and narratives can be
wrong or omit context. Raw evidence and narratives stay in session memory.
No report persistence or TB4-B functionality is implemented.

Aggregate statements show canonical metric provenance without individual issue
citations. Qualitative observations retain Python-resolved issue links; combined
findings keep statistical and issue support separate. Source-type metadata and
the narrative instructions distinguish reported bugs from feature requests.
Rejected claims can be inspected in a collapsed developer expander with bounded,
session-only diagnostics; no rejected text, excerpts or raw responses are saved
or logged. Filtering remains conservative and semantic support needs manual review.

The prototype is frozen for the team demonstration. See
[`docs/TB4A_HANDOFF.md`](docs/TB4A_HANDOFF.md) for verified budgets, live usage,
tests and remaining limitations. TB4-B requires separate approval.
