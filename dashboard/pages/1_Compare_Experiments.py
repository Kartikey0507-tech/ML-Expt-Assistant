import os

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Compare experiments",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Compare experiments")
st.caption("Compare two completed runs using their stored metrics and settings.")


def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


try:
    experiments = api_get("/experiments/")
except requests.RequestException as error:
    st.error(f"Could not reach the backend: {error}")
    st.stop()

completed_experiments = [
    experiment
    for experiment in experiments
    if experiment["status"] == "completed"
]

if len(completed_experiments) < 2:
    st.info("Run at least two completed experiments before comparing them.")
    st.stop()

experiment_options = [experiment["id"] for experiment in completed_experiments]


def experiment_label(experiment_id: str) -> str:
    experiment = next(
        item
        for item in completed_experiments
        if item["id"] == experiment_id
    )
    return f"{experiment['name']} — {experiment_id[:8]}"


left_id, right_id = st.columns(2)

with left_id:
    left_experiment_id = st.selectbox(
        "Baseline experiment",
        options=experiment_options,
        format_func=experiment_label,
        key="left_experiment",
    )

with right_id:
    right_experiment_id = st.selectbox(
        "Comparison experiment",
        options=experiment_options,
        index=1,
        format_func=experiment_label,
        key="right_experiment",
    )

if left_experiment_id == right_experiment_id:
    st.warning("Select two different experiments.")
    st.stop()

try:
    comparison = api_get(
        "/comparisons/"
        f"?left_experiment_id={left_experiment_id}"
        f"&right_experiment_id={right_experiment_id}"
    )
except requests.RequestException as error:
    st.error(f"Could not compare experiments: {error}")
    st.stop()

st.subheader("Result")
st.success(comparison["winner_reason"])

metric_one, metric_two = st.columns(2)

with metric_one:
    st.metric(
        "Accuracy difference",
        f"{comparison['accuracy_difference']:+.2%}",
        help="Right experiment accuracy minus baseline accuracy.",
    )

with metric_two:
    st.metric(
        "Loss difference",
        f"{comparison['loss_difference']:+.4f}",
        help="Right experiment loss minus baseline loss.",
    )

st.subheader("Final metrics")

metrics_frame = pd.DataFrame(
    [
        {
            "Experiment": comparison["left"]["name"],
            "Test accuracy": comparison["left"]["test_accuracy"],
            "Test loss": comparison["left"]["test_loss"],
        },
        {
            "Experiment": comparison["right"]["name"],
            "Test accuracy": comparison["right"]["test_accuracy"],
            "Test loss": comparison["right"]["test_loss"],
        },
    ]
).set_index("Experiment")

st.dataframe(metrics_frame, use_container_width=True)

st.subheader("Configuration differences")

if comparison["configuration_differences"]:
    configuration_frame = pd.DataFrame(
        [
            {
                "Setting": setting,
                "Baseline": values["left"],
                "Comparison": values["right"],
            }
            for setting, values in comparison["configuration_differences"].items()
        ]
    )
    st.dataframe(configuration_frame, use_container_width=True, hide_index=True)
else:
    st.info("The two experiments used identical configurations.")

st.subheader("Training curves")

try:
    left_metrics = api_get(
        f"/experiments/{left_experiment_id}/metrics"
    )
    right_metrics = api_get(
        f"/experiments/{right_experiment_id}/metrics"
    )

    left_frame = pd.DataFrame(left_metrics).set_index("epoch")
    right_frame = pd.DataFrame(right_metrics).set_index("epoch")

    loss_frame = pd.DataFrame(
        {
            f"{comparison['left']['name']} train loss": left_frame["train_loss"],
            f"{comparison['left']['name']} validation loss": (
                left_frame["validation_loss"]
            ),
            f"{comparison['right']['name']} train loss": right_frame["train_loss"],
            f"{comparison['right']['name']} validation loss": (
                right_frame["validation_loss"]
            ),
        }
    )

    accuracy_frame = pd.DataFrame(
        {
            f"{comparison['left']['name']} train accuracy": (
                left_frame["train_accuracy"]
            ),
            f"{comparison['left']['name']} validation accuracy": (
                left_frame["validation_accuracy"]
            ),
            f"{comparison['right']['name']} train accuracy": (
                right_frame["train_accuracy"]
            ),
            f"{comparison['right']['name']} validation accuracy": (
                right_frame["validation_accuracy"]
            ),
        }
    )

    chart_one, chart_two = st.columns(2)

    with chart_one:
        st.write("Loss")
        st.line_chart(loss_frame)

    with chart_two:
        st.write("Accuracy")
        st.line_chart(accuracy_frame)

except requests.RequestException as error:
    st.warning(f"Could not load curves: {error}")

st.subheader("Diagnostics")

diagnostic_one, diagnostic_two = st.columns(2)

with diagnostic_one:
    st.write(comparison["left"]["name"])
    if comparison["left"]["diagnostics"]:
        for finding in comparison["left"]["diagnostics"]:
            st.write(f"- {finding}")
    else:
        st.write("No stored diagnostic findings.")

with diagnostic_two:
    st.write(comparison["right"]["name"])
    if comparison["right"]["diagnostics"]:
        for finding in comparison["right"]["diagnostics"]:
            st.write(f"- {finding}")
    else:
        st.write("No stored diagnostic findings.")