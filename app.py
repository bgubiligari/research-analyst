"""Local TB1 dashboard; no persistent source cache or report files."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from research import ResearchError, run_research

load_dotenv()
st.set_page_config(page_title="Community Research Analyst")
st.title("Community Research Analyst — TB1")
st.caption("Up to 10 public GitHub issues, one classification batch, in-memory results.")
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
    for warning in result.warnings:
        st.warning(warning)
    st.write(f"GitHub reports {result.batch.total} matches; retrieved {len(result.batch.items)} eligible issues.")
    st.write(result.counts)
    st.subheader("Sampled GitHub issue sentiment")
    st.caption("Denominator: relevant classified issues with sentiment other than unclear.")
    st.dataframe(result.sentiment, hide_index=True)
    if result.sentiment.denominator.iloc[0]:
        st.bar_chart(result.sentiment.set_index("sentiment")["count"])
    st.subheader("Original evidence — temporary session data")
    labels = {label.source_id: label for label in result.labels}
    rows = [{"url": item.url, "source_id": item.source_id, "title": item.title,
             "created_at": item.created_at,
             "relevance": labels[item.source_id].relevance if item.source_id in labels else "unclassified",
             "sentiment": labels[item.source_id].sentiment if item.source_id in labels else "unclassified"}
            for item in result.batch.items]
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("Original GitHub issue")})
