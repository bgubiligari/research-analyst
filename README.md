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

Results exist only in active session memory; refresh/disconnect may lose them.
There is no source-text logging, persistent cache or report saving. `store=False`
is used for the model response; this does not promise zero provider-side retention.
GitHub issues overrepresent problems; this tiny non-random sample cannot describe
all developers or historical trends. Classification is model interpretation.

Offline evaluation runs synthetic fixtures through the same `run_research`
function the UI calls; tests require no credentials and do not enable a fake-live
mode in the app. TB2 is proposed only after explicit human review.
