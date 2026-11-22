import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Upload dataset", page_icon="📁")
st.title("📁 Upload a dataset")
st.caption(
    "Upload CSV files for future tabular/text experiments, or ZIP files "
    "for future image-classification experiments."
)

dataset_name = st.text_input(
    "Dataset name",
    placeholder="Example: customer-churn-data",
)

uploaded_file = st.file_uploader(
    "Choose a CSV or ZIP file",
    type=["csv", "zip"],
)

if st.button("Upload and validate dataset"):
    if not dataset_name.strip():
        st.warning("Enter a dataset name.")

    elif uploaded_file is None:
        st.warning("Choose a CSV or ZIP file.")

    else:
        try:
            response = requests.post(
                f"{API_BASE_URL}/datasets/upload",
                data={"name": dataset_name.strip()},
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                },
                timeout=120,
            )
            response.raise_for_status()

            dataset = response.json()

            if dataset["status"] == "validated":
                st.success("Dataset uploaded and validated.")
            else:
                st.error("Dataset uploaded but did not pass validation.")

            st.json(dataset["validation"])

        except requests.RequestException as error:
            st.error(f"Could not upload dataset: {error}")

st.divider()
st.subheader("Registered datasets")

try:
    response = requests.get(f"{API_BASE_URL}/datasets/", timeout=15)
    response.raise_for_status()
    datasets = response.json()

    if not datasets:
        st.info("No uploaded datasets yet.")
    else:
        st.dataframe(
            [
                {
                    "Name": dataset["name"],
                    "File": dataset["original_filename"],
                    "Type": dataset["file_type"],
                    "Size (bytes)": dataset["size_bytes"],
                    "Status": dataset["status"],
                    "Created": dataset["created_at"],
                }
                for dataset in datasets
            ],
            use_container_width=True,
            hide_index=True,
        )

except requests.RequestException as error:
    st.warning(f"Could not load registered datasets: {error}")