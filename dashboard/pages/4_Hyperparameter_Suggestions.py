import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Hyperparameter suggestions",
    page_icon="💡",
)

st.title("💡 Hyperparameter suggestions")
st.caption(
    "Recommendations are derived from deterministic diagnostics. "
    "You approve every new experiment yourself."
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
    st.info("Complete an experiment before requesting suggestions.")
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

if st.button("Generate deterministic suggestions"):
    try:
        response = requests.get(
            f"{API_BASE_URL}/experiments/{selected_id}/suggestions",
            timeout=15,
        )
        response.raise_for_status()
        st.session_state["suggestions"] = response.json()

    except requests.RequestException as error:
        st.error(f"Could not generate suggestions: {error}")

result = st.session_state.get("suggestions")

if result and result["experiment_id"] == selected_id:
    st.subheader("Suggested next experiments")

    for suggestion in result["suggestions"]:
        with st.container(border=True):
            st.write(f"### {suggestion['parameter']}")
            st.write(
                f"**Current:** `{suggestion['current_value']}`  \n"
                f"**Suggested:** `{suggestion['suggested_value']}`"
            )
            st.write(suggestion["reason"])

            if suggestion["evidence_used"]:
                st.caption(
                    "Evidence fields: "
                    + ", ".join(suggestion["evidence_used"])
                )