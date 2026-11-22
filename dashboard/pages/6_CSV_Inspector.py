import os

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="CSV Inspector", page_icon="🔎", layout="wide")

st.title("CSV Inspector")
st.write(
    "Inspect an uploaded CSV before choosing its target column and training a model."
)


def load_datasets():
    response = requests.get(f"{API_BASE_URL}/datasets/", timeout=20)
    response.raise_for_status()
    return response.json()


try:
    all_datasets = load_datasets()
    csv_datasets = [
        dataset
        for dataset in all_datasets
        if dataset["file_type"] == "csv"
    ]
except requests.RequestException as error:
    st.error(f"Could not load datasets: {error}")
    st.stop()

if not csv_datasets:
    st.info("No CSV datasets have been uploaded yet. Go to Upload Dataset first.")
    st.stop()

dataset_map = {
    f'{dataset["name"]} — {dataset["original_filename"]}': dataset
    for dataset in csv_datasets
}

selected_label = st.selectbox(
    "Choose a CSV dataset",
    list(dataset_map.keys()),
)

selected_dataset = dataset_map[selected_label]

st.caption(f'Dataset ID: `{selected_dataset["id"]}`')

if st.button("Inspect CSV", type="primary"):
    try:
        response = requests.get(
            f'{API_BASE_URL}/datasets/{selected_dataset["id"]}/profile',
            timeout=30,
        )
        response.raise_for_status()
        st.session_state["csv_profile"] = response.json()
        st.session_state["profile_dataset_id"] = selected_dataset["id"]
    except requests.RequestException as error:
        st.error(f"Could not inspect this CSV: {error}")

profile = st.session_state.get("csv_profile")

if (
    profile
    and st.session_state.get("profile_dataset_id") == selected_dataset["id"]
):
    st.divider()
    st.subheader("Dataset overview")

    left, middle, right = st.columns(3)
    left.metric("Rows sampled", profile["sampled_rows"])
    middle.metric("Columns", profile["column_count"])
    right.metric("Encoding", profile["encoding"])

    st.info(profile["note"])

    st.subheader("Column profile")

    column_table = pd.DataFrame(profile["columns"])
    st.dataframe(column_table, use_container_width=True, hide_index=True)

    st.subheader("Data preview")
    st.dataframe(
        pd.DataFrame(profile["preview"]),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Do not train yet. In the next step, you will select the target column "
        "and choose classification or regression."
    )