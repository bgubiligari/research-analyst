"""Local TB1 dashboard; no persistent source cache or report files."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv

from research import (
    ResearchError,
    evidence_table,
    ranked_complaints,
    research_findings,
    run_research,
)

load_dotenv()
st.set_page_config(page_title="Community Research Analyst", layout="wide")
st.title("Community Research Analyst — TB1")
st.caption("Up to 10 public GitHub issues, one classification batch, in-memory results.")
st.info(
    "Retrieval scope: a small repository-level sample of the newest 10 public issues "
    "within your selected date range. Your question guides classification only; it "
    "does not filter the GitHub search. Narrow topics may have few or no relevant issues."
)
with st.form("research"):
    question = st.text_area("Research question", "What complaints appear in recent LangChain GitHub Issues?")
    repo = st.text_input("Repository", "langchain-ai/langchain")
    start = st.date_input("From", datetime.now(ZoneInfo("America/Los_Angeles")).date() - timedelta(days=30))
    end = st.date_input("Through", datetime.now(ZoneInfo("America/Los_Angeles")).date())
    submitted = st.form_submit_button("Research")
if submitted:
    st.session_state.pop("active_result", None)
    try:
        with st.status("Research running", expanded=True) as status:
            result = run_research(question, repo.strip(), start, end, progress=st.write)
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
    for warning in result.warnings:
        if warning not in sampling_notes:
            st.warning(warning)
    cards = st.columns(4)
    for card, title, value in zip(cards,
            ["GitHub reported matches", "Issues retrieved", "Relevant issues", "Classified issues"],
            [result.batch.total, result.counts["collected"], result.counts["relevant"],
             result.counts["classified"]], strict=True):
        card.metric(title, value)
    st.caption(
        f"{result.counts['collected']} of {result.batch.total} matching GitHub issues were retrieved. "
        "This is a small, non-random sample of the newest issues, not a representative "
        "sample of all developers."
    )
    with st.expander("Detailed classification counts"):
        st.write(result.counts)
    if not hasattr(result, "complaints"):
        st.info("Run Research again to calculate complaint themes for this session.")
        st.stop()
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
