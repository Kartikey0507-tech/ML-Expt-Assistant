import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Download report", page_icon="📄")
st.title("📄 Download experiment report")
st.caption(
    "Create a Markdown report from tracked metrics, diagnostics, "
    "and any saved Gemini explanation."
)

try:
    response = requests.get(f"{API_BASE_URL}/experiments/", timeout=15)
    response.raise_for_status()
    experiments = response.json()
except requests.RequestException as error:
    st.error(f"Could not load experiments: {error}")
    st.stop()

completed_experiments = [
    experiment
    for experiment in experiments
    if experiment["status"] == "completed"
]

if not completed_experiments:
    st.info("Complete an experiment before generating a report.")
    st.stop()

selected_id = st.selectbox(
    "Choose a completed experiment",
    options=[experiment["id"] for experiment in completed_experiments],
    format_func=lambda experiment_id: next(
        f"{experiment['name']} — {experiment_id[:8]}"
        for experiment in completed_experiments
        if experiment["id"] == experiment_id
    ),
)

if st.button("Prepare Markdown report"):
    try:
        report_response = requests.get(
            f"{API_BASE_URL}/experiments/{selected_id}/report.md",
            timeout=20,
        )
        report_response.raise_for_status()

        st.download_button(
            label="Download report",
            data=report_response.text,
            file_name=f"experiment-{selected_id[:8]}-report.md",
            mime="text/markdown",
        )

        st.success("Report is ready to download.")

    except requests.RequestException as error:
        st.error(f"Could not create report: {error}")