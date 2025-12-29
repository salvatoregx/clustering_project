import streamlit as st
import json
import os
import glob
from .. import config

def show():
    """Displays the Data Integrity tab with Great Expectations validation results."""
    st.header("Data Integrity Checks (Great Expectations)")

    json_files = glob.glob(os.path.join(config.VALIDATION_PATH, "*.json"))
    if not json_files:
        st.warning("No validation results found. Please run the ETL pipeline first.")
        return

    json_files.sort(key=os.path.getmtime, reverse=True)
    selected_file = st.selectbox("Select Validation Run", json_files, format_func=lambda x: os.path.basename(x))

    if not selected_file:
        return

    with open(selected_file, 'r') as f:
        data = json.load(f)

    success = data.get("success", False)
    status_message = "Validation Passed" if success else "Validation Failed"
    st.metric("Overall Status", status_message)

    results = data.get("results", [])
    for res in results:
        expectation_config = res.get("expectation_config", {})
        expectation_type = expectation_config.get("expectation_type", "Unknown")
        kwargs = expectation_config.get("kwargs", {})
        
        with st.expander(f"{expectation_type}"):
            st.write(f"**Column:** `{kwargs.get('column', 'N/A')}`")
            st.write(f"**Parameters:** `{kwargs}`")
            st.write(f"**Observed Value:** `{res.get('result', {}).get('observed_value')}`")
            st.success("Success") if res.get("success") else st.error("Failed")