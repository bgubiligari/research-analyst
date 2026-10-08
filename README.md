# Community Research Analyst — Tracer Bullet 1

Local synchronous Streamlit research using public GitHub Issues, typed OpenAI
classification, and deterministic Pandas sentiment counts. No saved reports,
trends, Deep mode, or synthetic live data.

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
The default window is 30 days. One newest-first, unfiltered repository/date search
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
all developers or historical trends. Classification is model interpretation.

Offline evaluation runs synthetic fixtures through the same `run_research`
function the UI calls; tests require no credentials and do not enable a fake-live
mode in the app. TB2 is proposed only after explicit human review.
