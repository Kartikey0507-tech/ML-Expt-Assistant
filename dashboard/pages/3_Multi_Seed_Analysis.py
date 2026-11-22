import os

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Multi-seed analysis",
    page_icon="🎲",
    layout="wide",
)

st.title("🎲 Multi-seed statistical analysis")
st.caption(
    "Select repeated runs with the same settings and different random seeds."
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

if len(completed_experiments) < 2:
    st.info("Complete at least two experiments before using this page.")
    st.stop()


def experiment_label(experiment_id: str) -> str:
    experiment = next(
        item
        for item in completed_experiments
        if item["id"] == experiment_id
    )

    return (
        f"{experiment['name']} — seed "
        f"{experiment['config']['seed']} — {experiment_id[:8]}"
    )


selected_ids = st.multiselect(
    "Select completed runs",
    options=[experiment["id"] for experiment in completed_experiments],
    format_func=experiment_label,
)

if len(selected_ids) < 2:
    st.info("Select at least two completed runs.")
    st.stop()

if st.button("Analyze selected seeds"):
    try:
        response = requests.post(
            f"{API_BASE_URL}/seed-analysis/",
            json={"experiment_ids": selected_ids},
            timeout=20,
        )
        response.raise_for_status()
        analysis = response.json()

        st.session_state["seed_analysis"] = analysis

    except requests.RequestException as error:
        st.error(f"Could not analyze selected runs: {error}")

analysis = st.session_state.get("seed_analysis")

if analysis:
    st.subheader("Aggregate result")

    accuracy_column, loss_column, count_column = st.columns(3)

    with accuracy_column:
        st.metric(
            "Mean test accuracy",
            f"{analysis['mean_test_accuracy']:.2%}",
        )
        st.caption(
            "Standard deviation: "
            f"{analysis['accuracy_standard_deviation']:.2%}"
        )

    with loss_column:
        st.metric(
            "Mean test loss",
            f"{analysis['mean_test_loss']:.4f}",
        )
        st.caption(
            "Standard deviation: "
            f"{analysis['loss_standard_deviation']:.4f}"
        )

    with count_column:
        st.metric("Runs analyzed", analysis["run_count"])

    st.subheader("Individual seed results")

    results_frame = pd.DataFrame(analysis["runs"]).set_index("seed")
    results_frame = results_frame[
        ["name", "test_accuracy", "test_loss", "experiment_id"]
    ]

    st.dataframe(results_frame, use_container_width=True)
    st.bar_chart(results_frame[["test_accuracy"]])

    if analysis["configuration_mismatches"]:
        st.warning(
            "Some selected runs differ in settings other than the seed. "
            "Their aggregate statistics are less reliable."
        )
        st.json(analysis["configuration_mismatches"])
    else:
        st.success(
            "All selected runs have identical settings except for seed."
        )