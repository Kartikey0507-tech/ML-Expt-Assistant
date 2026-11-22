import os

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="ML Experiment Assistant",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Intelligent ML Experiment Assistant")
st.caption("Run, evaluate, diagnose, and explain Fashion-MNIST experiments.")


def get_json(url: str):
    """Fetch JSON; return None when the API reports that data is absent."""
    response = requests.get(url, timeout=15)

    if response.status_code == 404:
        return None

    response.raise_for_status()
    return response.json()


# -------------------------------------------------------------------
# Create experiment
# -------------------------------------------------------------------
st.subheader("Create an experiment")

with st.form("create_experiment"):
    name = st.text_input("Experiment name", value="fashion-mnist-baseline")

    left_column, right_column = st.columns(2)

    with left_column:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=100,
            value=2,
        )

        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.00001,
            max_value=1.0,
            value=0.001,
            format="%.5f",
        )

    with right_column:
        batch_size = st.selectbox(
            "Batch size",
            [16, 32, 64, 128],
            index=1,
        )

        seed = st.number_input(
            "Random seed",
            min_value=0,
            value=42,
        )

        early_stopping_patience = st.number_input(
            "Early stopping patience",
            min_value=1,
            max_value=20,
            value=3,
            help=(
                "Stop training after this many epochs without "
                "validation-loss improvement."
            ),
        )

    submitted = st.form_submit_button("Create queued experiment")

if submitted:
    payload = {
        "name": name,
        "dataset_name": "fashion_mnist",
        "model_name": "simple_cnn",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "early_stopping_patience": int(early_stopping_patience),
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/experiments/",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        st.success("Experiment created successfully.")

    except requests.RequestException as error:
        st.error(f"Could not create experiment: {error}")


# -------------------------------------------------------------------
# Load experiments
# -------------------------------------------------------------------
st.divider()
st.subheader("Experiments")

if st.button("Refresh experiment data"):
    st.rerun()

try:
    experiments = get_json(f"{API_BASE_URL}/experiments/")

except requests.RequestException as error:
    st.error(f"Could not reach the FastAPI backend: {error}")
    st.stop()

if not experiments:
    st.info("No experiments yet. Create your first experiment above.")
    st.stop()

st.dataframe(
    [
        {
            "Name": experiment["name"],
            "Status": experiment["status"],
            "Epochs": experiment["config"]["epochs"],
            "Learning rate": experiment["config"]["learning_rate"],
            "Created": experiment["created_at"],
        }
        for experiment in experiments
    ],
    use_container_width=True,
    hide_index=True,
)

selected_id = st.selectbox(
    "Select an experiment",
    options=[experiment["id"] for experiment in experiments],
    format_func=lambda experiment_id: next(
        f"{experiment['name']} — {experiment['status']}"
        for experiment in experiments
        if experiment["id"] == experiment_id
    ),
)

selected_experiment = next(
    experiment
    for experiment in experiments
    if experiment["id"] == selected_id
)

status_column, action_column = st.columns(2)

with status_column:
    st.write(f"Current status: **{selected_experiment['status']}**")

with action_column:
    if st.button("Start selected experiment"):
        if selected_experiment["status"] != "queued":
            st.warning("Only queued experiments can be started.")

        else:
            try:
                response = requests.post(
                    f"{API_BASE_URL}/experiments/{selected_id}/start",
                    timeout=15,
                )
                response.raise_for_status()
                st.success("Training started in the background.")

            except requests.RequestException as error:
                st.error(f"Could not start training: {error}")

if selected_experiment["error_message"]:
    st.error(f"Worker error: {selected_experiment['error_message']}")


# -------------------------------------------------------------------
# Training curves and metrics
# -------------------------------------------------------------------
st.divider()
st.subheader(f"Run details: {selected_experiment['name']}")

try:
    metrics = get_json(
        f"{API_BASE_URL}/experiments/{selected_id}/metrics"
    )

    if metrics:
        metrics_frame = pd.DataFrame(metrics).set_index("epoch")

        loss_column, accuracy_column = st.columns(2)

        with loss_column:
            st.write("Loss curves")
            st.line_chart(
                metrics_frame[["train_loss", "validation_loss"]]
            )

        with accuracy_column:
            st.write("Accuracy curves")
            st.line_chart(
                metrics_frame[["train_accuracy", "validation_accuracy"]]
            )

        st.write("Per-epoch metrics")
        st.dataframe(metrics_frame, use_container_width=True)

    elif selected_experiment["status"] == "queued":
        st.info("This experiment is waiting to start.")

    elif selected_experiment["status"] == "running":
        st.info("Training is in progress. Refresh shortly to see metrics.")

except requests.RequestException as error:
    st.warning(f"Could not load metrics: {error}")


# -------------------------------------------------------------------
# Completed-run results
# -------------------------------------------------------------------
if selected_experiment["status"] == "completed":
    st.divider()
    st.subheader("Final evaluation")

    try:
        evaluation = get_json(
            f"{API_BASE_URL}/experiments/{selected_id}/evaluation"
        )

        if evaluation is None:
            st.info(
                "This older experiment has no stored final evaluation."
            )

        else:
            test_accuracy_column, test_loss_column = st.columns(2)

            with test_accuracy_column:
                st.metric(
                    "Test accuracy",
                    f"{evaluation['test_accuracy']:.2%}",
                )

            with test_loss_column:
                st.metric(
                    "Test loss",
                    f"{evaluation['test_loss']:.4f}",
                )

    except requests.RequestException as error:
        st.warning(f"Could not load final evaluation: {error}")

    # ---------------------------------------------------------------
    # Confusion matrix and class metrics
    # ---------------------------------------------------------------
    try:
        classification = get_json(
            f"{API_BASE_URL}/experiments/{selected_id}/classification"
        )

        if classification is None:
            st.info(
                "This older experiment has no classification evaluation."
            )

        else:
            st.divider()
            st.subheader("Confusion matrix")

            class_names = classification["class_names"]

            matrix_frame = pd.DataFrame(
                classification["confusion_matrix"],
                index=class_names,
                columns=class_names,
            )

            st.caption("Rows are actual classes; columns are predicted classes.")
            st.dataframe(matrix_frame, use_container_width=True)

            st.subheader("Per-class performance")

            class_metrics_frame = pd.DataFrame(
                classification["class_metrics"]
            ).set_index("class_name")

            st.bar_chart(
                class_metrics_frame[
                    ["precision", "recall", "f1_score"]
                ]
            )

            st.dataframe(
                class_metrics_frame,
                use_container_width=True,
            )

            st.subheader("Most confused class pairs")

            confused_pairs_frame = pd.DataFrame(
                classification["most_confused_pairs"]
            )

            st.dataframe(
                confused_pairs_frame,
                use_container_width=True,
            )

    except requests.RequestException as error:
        st.warning(f"Could not load classification results: {error}")

    # ---------------------------------------------------------------
    # Deterministic diagnostics
    # ---------------------------------------------------------------
    try:
        diagnostics = get_json(
            f"{API_BASE_URL}/experiments/{selected_id}/diagnostics"
        )

        st.divider()
        st.subheader("Deterministic diagnostics")

        if diagnostics is None:
            st.info(
                "This older experiment has no diagnostic findings."
            )

        elif not diagnostics:
            st.success("No significant diagnostic issues were detected.")

        else:
            for finding in diagnostics:
                heading = (
                    f"[{finding['severity'].upper()}] "
                    f"{finding['title']}"
                )

                with st.expander(heading):
                    st.write("Evidence")
                    st.json(finding["evidence"])

                    st.write("Recommended actions")
                    for recommendation in finding["recommendations"]:
                        st.write(f"- {recommendation}")

                    st.caption(
                        f"Rule version: {finding['rule_version']}"
                    )

    except requests.RequestException as error:
        st.warning(f"Could not load diagnostics: {error}")

    # ---------------------------------------------------------------
    # Manually triggered Gemini explanation
    # ---------------------------------------------------------------
    st.divider()
    st.subheader("Grounded Gemini explanation")

    if st.button("Generate explanation with Gemini"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/experiments/{selected_id}/explanation",
                timeout=60,
            )
            response.raise_for_status()
            st.success("Grounded Gemini explanation generated.")
            st.rerun()

        except requests.RequestException as error:
            st.error(f"Could not generate explanation: {error}")

    try:
        explanation = get_json(
            f"{API_BASE_URL}/experiments/{selected_id}/explanation"
        )

        if explanation is None:
            st.info(
                "No Gemini explanation exists yet. "
                "Click the button above when you want one."
            )

        else:
            st.write(explanation["summary"])

            st.caption(
                f"Provider: {explanation['provider']} | "
                f"Model: {explanation['model']} | "
                f"Grounding passed: {explanation['grounding_passed']}"
            )

            st.write("Recommended actions")

            for action in explanation["recommended_actions"]:
                st.write(f"- **{action['action']}**")
                st.caption(action["rationale"])

            st.write("Evidence used")
            st.code(", ".join(explanation["evidence_used"]))

            if explanation["limitations"]:
                st.write("Limitations")

                for limitation in explanation["limitations"]:
                    st.write(f"- {limitation}")

    except requests.RequestException as error:
        st.warning(f"Could not load Gemini explanation: {error}")