"""Local TB1/TB2 dashboard; no persistent source cache or report files."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv

from research import (
    HISTORICAL_BIAS,
    ResearchError,
    completed_month_range,
    evidence_table,
    historical_chart_data,
    ranked_complaints,
    research_findings,
    run_research,
)

load_dotenv()
st.set_page_config(page_title="Community Research Analyst", layout="wide")
st.title("Community Research Analyst — TB2")
st.caption("Historical: up to 50 issues, 12 searches, five classification batches; in-memory results.")
historical = st.radio("Sampling", ["Historical (TB2)", "Recent (TB1)"], horizontal=True) == "Historical (TB2)"
st.info(
    ("Retrieval scope: up to 50 public issues across calendar months. " if historical else
     "Retrieval scope: a small repository-level sample of the newest 10 public issues ") +
    "within your selected date range. Your question guides classification only; it "
    "does not filter the GitHub search. Narrow topics may have few or no relevant issues."
)
default_start, default_end = completed_month_range() if historical else (
    datetime.now(ZoneInfo("America/Los_Angeles")).date() - timedelta(days=30),
    datetime.now(ZoneInfo("America/Los_Angeles")).date())
if historical:
    st.caption("Default: six completed UTC calendar months. Early halves use ascending creation order; late halves descending. Fixed quotas, no refill.")
with st.form("research"):
    question = st.text_area("Research question",
        "How has sentiment about LangChain changed in sampled GitHub Issues over these six months?"
        if historical else "What complaints appear in recent LangChain GitHub Issues?")
    repo = st.text_input("Repository", "langchain-ai/langchain")
    start = st.date_input("From", default_start, key=f"start_{historical}")
    end = st.date_input("Through", default_end, key=f"end_{historical}")
    submitted = st.form_submit_button("Research")
if submitted:
    st.session_state.pop("active_result", None)
    try:
        with st.status("Research running", expanded=True) as status:
            result = run_research(question, repo.strip(), start, end, progress=st.write, historical=historical)
            st.session_state.active_result = (repo.strip(), start, end, result)
            status.update(label="Research finished — inspect coverage and warnings", state="complete")
    except ResearchError as exc:
        st.error(str(exc))
if "active_result" in st.session_state:
    repo, start, end, result = st.session_state.active_result
    st.write(f"Scope: {repo}, {start} through {end}; source: public GitHub Issues.")
    sampling_notes = {
        "Small exploratory sample of GitHub Issues; not overall developer sentiment.",
        "Newest issues first; this is a bounded, non-random sample, not a trend.",
    }
    hidden_warnings = {
        HISTORICAL_BIAS,
        "Monthly sample sizes and coverage may be uneven; changes describe sampled issues only.",
    }
    for warning in result.warnings:
        if warning not in sampling_notes | hidden_warnings:
            st.warning(warning)
    cards = st.columns(4)
    for card, title, value in zip(cards,
            ["GitHub reported matches", "Issues retrieved", "Relevant issues", "Classified issues"],
            [result.batch.total if result.batch.total is not None else "N/A", result.counts["collected"], result.counts["relevant"],
             result.counts["classified"]], strict=True):
        card.metric(title, value)
    st.caption(
        f"{result.counts['collected']} of {result.batch.total if result.batch.total is not None else "N/A"} matching GitHub issues were retrieved. "
        "This is a small, non-random sample of issues, not a representative "
        "sample of all developers."
    )
    with st.expander("Detailed classification counts"):
        st.write(result.counts)
    if not hasattr(result, "complaints"):
        st.info("Run Research again to calculate complaint themes for this session.")
        st.stop()
    if not result.monthly.empty:
        st.subheader("Historical Sentiment in Sampled GitHub Issues")
        st.caption("Percentages use relevant classified issues excluding unclear sentiment. N < 5 is suppressed; gaps are not connected. Missing values are N/A.")
        trend = historical_chart_data(result.monthly)
        if trend.empty:
            st.info("Insufficient monthly sentiment observations for a trend chart.")
        else:
            st.vega_lite_chart(trend, {
                "mark": {"type": "line", "point": True},
                "encoding": {
                    "x": {"field": "month", "type": "ordinal", "sort": result.monthly.month.tolist(),
                          "scale": {"domain": result.monthly.month.tolist()}},
                    "y": {"field": "percent", "type": "quantitative", "title": "Sample sentiment (%)",
                          "scale": {"domain": [0, 100]}},
                    "color": {"field": "sentiment", "type": "nominal"},
                    "detail": {"field": "series", "type": "nominal"},
                    "tooltip": [{"field": field} for field in ["month", "sentiment", "count", "denominator", "percent"]],
                },
            }, width="stretch")
        st.subheader("Monthly Coverage and Denominators")
        st.caption("Hover over a column heading for its definition. Scroll horizontally to see all columns.")
        coverage_help = {
            "month": "UTC calendar month, based on when each issue was created.",
            "start": "First included creation date for this month (UTC, inclusive).",
            "end": "Last included creation date for this month (UTC, inclusive).",
            "partial": "True when the requested dates cover only part of this calendar month.",
            "reported_matches": "Matching issues reported by GitHub across this month's disjoint searches, before sampling. N/A if any search total is unknown; incomplete searches may undercount.",
            "quota": "Maximum issues allocated to this month before retrieval. Unused capacity is not refilled or moved to other months.",
            "returned": "Records returned by successful HTTP responses, before validation and deduplication. These are not necessarily accepted issues.",
            "retrieved": "Unique eligible issues accepted into this month's sample after validation and deduplication.",
            "rejected": "Inspected records discarded as invalid, duplicate, pull requests, outside the requested dates, or over the issue cap.",
            "classified": "Retrieved issues with a valid, uniquely matched model classification, regardless of relevance.",
            "relevant": "Classified issues considered relevant to your question, including those with unclear sentiment.",
            "unclassified": "Retrieved issues without an accepted classification, including failed, missing, invalid, or duplicate model outputs.",
            "irrelevant": "Classified issues considered unrelated to your question. Excluded from sentiment counts and percentages.",
            "uncertain": "Classified issues whose relevance could not be determined. Excluded from sentiment counts and percentages.",
            "unclear": "Relevant classified issues whose sentiment could not be determined. Excluded from the sentiment denominator.",
            "denominator": "Number of relevant classified issues with sentiment other than unclear. All sentiment percentages use this N; N = 0 gives N/A, and N < 5 is hidden on the trend chart.",
            "incomplete": "True if GitHub flagged at least one search response as incomplete. Check errors and unattempted too: False alone does not guarantee complete coverage.",
            "errors": "Number of attempted search intervals that failed. Earlier successful results are retained.",
            "unattempted": "Number of search intervals not requested because rate limits or the retrieval time budget stopped collection.",
        }
        for sentiment in ("positive", "negative", "neutral", "mixed"):
            coverage_help[sentiment] = f"Number of relevant classified issues labeled {sentiment}. Each issue contributes to one sentiment category."
            coverage_help[f"{sentiment}_percent"] = f"{sentiment.title()} count divided by the sentiment denominator, multiplied by 100. N/A when the denominator is zero; small-N values remain in this table but are suppressed on the trend chart."
        st.dataframe(result.monthly, hide_index=True, column_config={
            column: st.column_config.Column(help=description)
            for column, description in coverage_help.items()
        })
        with st.expander("Search interval coverage and errors"):
            st.dataframe(result.intervals, hide_index=True)
    st.subheader("Research Findings")
    for finding in research_findings(result):
        st.write(finding)
    st.caption("These findings describe only the retrieved GitHub Issues sample.")

    st.subheader("Complaint Themes in Sampled GitHub Issues")
    st.caption(
        "Prevalence = issues mentioning a theme / all relevant classified issues, "
        "including those with no complaints or unclear sentiment. Each issue counts "
        "once per theme; multiple themes mean percentages may sum above 100%."
    )
    complaints = ranked_complaints(result).copy()
    if not complaints.empty:
        complaints["category"] = complaints.theme.str.replace("_", " ").str.title()
        complaints["label"] = complaints.percent.map(lambda value: f"{value:.1f}%")
        st.vega_lite_chart(complaints, {
            "autosize": {"type": "fit", "contains": "padding"},
            "height": max(130, min(280, len(complaints) * 32 + 45)),
            "padding": {"left": 5, "right": 10, "top": 5, "bottom": 5},
            "encoding": {
                "x": {"field": "percent", "type": "quantitative",
                      "title": "Relevant classified issues (%)",
                      "scale": {"domain": [0, 100], "nice": False, "zero": True},
                      "axis": {"values": [0, 20, 40, 60, 80, 100],
                               "labelExpr": "datum.value + '%'"}},
                "y": {"field": "category", "type": "nominal", "title": None,
                      "sort": complaints.category.tolist(), "axis": {"labelLimit": 140}},
                "tooltip": [
                    {"field": "category", "type": "nominal", "title": "Theme"},
                    {"field": "count", "type": "quantitative", "title": "Issues"},
                    {"field": "denominator", "type": "quantitative", "title": "Denominator"},
                    {"field": "label", "type": "nominal", "title": "Prevalence"},
                ],
            },
            "layer": [
                {"mark": {"type": "bar", "color": "#1769aa"}},
                # Small bars need outside labels; large bars keep labels inside
                # so even a 100% bar fits without clipping its percentage.
                {"transform": [{"filter": "datum.percent < 15"}],
                 "mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 4},
                 "encoding": {"text": {"field": "label"}}},
                {"transform": [{"filter": "datum.percent >= 15"}],
                 "mark": {"type": "text", "align": "right", "baseline": "middle",
                          "dx": -5, "color": "white"},
                 "encoding": {"text": {"field": "label"}}},
            ],
        }, width="stretch")
    elif result.counts["relevant"]:
        st.info("No complaint themes were identified in the relevant classified issues.")
    else:
        st.info("No relevant classified issues; complaint prevalence is N/A.")
    with st.expander("Complaint counts and prevalence — all categories"):
        st.dataframe(result.complaints, hide_index=True,
                     column_config={"percent": st.column_config.NumberColumn("Prevalence (%)", format="%.1f")})

    st.subheader("Sentiment in Sampled GitHub Issues")
    st.caption("Denominator: relevant classified issues with sentiment other than unclear.")
    sentiment_counts = result.sentiment[["sentiment", "count"]].copy()
    sentiment_counts.loc[len(sentiment_counts)] = ["unclear", result.counts["unclear"]]
    if result.counts["relevant"]:
        st.bar_chart(sentiment_counts, x="sentiment", y="count", sort=False,
                     height=260,
                     x_label="Sentiment", y_label="Issue count")
    else:
        st.info("No classifiable relevant sentiment; chart unavailable.")
    st.caption(f"Unclear sentiment: {result.counts['unclear']} relevant issues (excluded from percentages).")
    with st.expander("Sentiment counts and percentages — all categories"):
        st.dataframe(result.sentiment, hide_index=True)

    st.subheader("Evidence Explorer")
    st.caption("Filters affect only this table; findings and charts retain the full retrieved sample.")
    filters = st.columns(2)
    sentiment_filter = filters[0].selectbox("Filter by sentiment",
        ["All", "positive", "negative", "neutral", "mixed", "unclear", "unclassified"])
    theme_filter = filters[1].selectbox("Filter by complaint theme",
        ["All"] + sorted({theme for label in result.labels for theme in label.complaint_themes}))
    evidence = evidence_table(result, sentiment=sentiment_filter, theme=theme_filter)
    st.caption(f"Showing {len(evidence)} of {len(result.batch.items)} retrieved issues.")
    st.dataframe(evidence, hide_index=True, width="stretch", row_height=48,
        height=48 * max(len(evidence), 1) + 38,
        column_config={
            "Issue": st.column_config.LinkColumn("Issue", display_text=r"/issues/(\d+)$", width=60),
            "Title": st.column_config.TextColumn("Title", width=300),
            "Sentiment": st.column_config.TextColumn("Sentiment", width=90),
            "Complaint Themes": st.column_config.TextColumn("Complaint Themes", width=220),
            "Relevance": st.column_config.TextColumn("Relevance", width=90),
            "Created At": st.column_config.DateColumn("Created At", format="YYYY-MM-DD", width=110),
        })
    if result.batch.items:
        with st.expander("Inspect original issue description"):
            selected_id = st.selectbox("Issue to inspect",
                [issue.source_id for issue in result.batch.items],
                format_func=lambda source_id: "#" + source_id.rsplit(":", 1)[-1])
            selected_issue = next(issue for issue in result.batch.items if issue.source_id == selected_id)
            st.link_button(f"Open original GitHub issue #{selected_issue.number}", selected_issue.url)
            st.text(selected_issue.title)
            st.caption(
                "Retained source text from this run, in session memory only. "
                "Descriptions are capped at 5,500 characters; use the original link for the full issue."
            )
            st.text(selected_issue.body or "No description was provided.")
    st.subheader("Notes")
    for warning in result.warnings:
        if warning in sampling_notes:
            st.caption(warning)
