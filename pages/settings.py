import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
from tools.memory import MemoryManager
from tools.paths import get_env_path

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="centered")
st.title("⚙️ Settings")
st.markdown("Adjust the settings for the various agents below.")

general, experiment, cache = st.tabs(
    ["General", "Experiment", "Cache"]
)

with general:

    # Changing API Key
    st.markdown("##### API Key Configuration")

    # Track editing mode
    editing = memory.get_var("editing", False)

    if memory.get_var("api_key") and not editing:
        st.success("Your API key was loaded successfully.")

        if st.button("Edit API Key"):
            memory.set_var("editing", True)
            if "api_key_input" in st.session_state:
                del st.session_state.api_key_input
            st.rerun()

    else:
        if memory.get_var("api_key_source") == "environment":
            st.info("Currently using API key from environment variables. Click 'Edit API Key' to set a custom one.")
        elif memory.get_var("api_key_source") == "secrets":
            st.info("Currently using API key from Streamlit secrets. Click 'Edit API Key' to set a custom one.")

        api_key_input = st.text_input(
            "Google Gemini API Key:",
            value="" if editing else memory.get_var("api_key", ""),
            type="password",
            help="Enter your Google Gemini API key. It will be saved securely and persist across page navigations.",
            key="api_key_input",
            placeholder="Enter your API key here..." if editing else None,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save API Key", use_container_width=True):
                if api_key_input and api_key_input.strip():
                    api_key = api_key_input.strip()
                    memory.set_var("api_key", api_key)
                    memory.set_var("api_key_source", "user")
                    memory.set_var("editing", False)
                    os.environ["GEMINI_API_KEY"] = api_key
                    os.environ["GOOGLE_API_KEY"] = api_key
                    try:
                        env_path = get_env_path()
                        with open(env_path, "w") as f:
                            f.write(f"GEMINI_API_KEY={api_key}\n")
                    except Exception:
                        pass
                    st.success("Your API key has been saved successfully!")
                    st.rerun()
                else:
                    st.error("Please enter your API key and try again.")

        with col2:
            if st.button("Cancel", use_container_width=True):
                memory.set_var("editing", False)
                st.rerun()

with experiment:
    st.markdown("##### Experiment Configuration")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        st.markdown("**Jupyter Server Configuration**")
        st.info("Configure once - used by all agents (Experiments, Curve Fitting, etc.)")

        jupyter_config = memory.get_var("jupyter_config", {})
        jupyter_url = st.text_input(
            "Jupyter Server URL:",
            value=jupyter_config.get("server_url", ""),
            help="Base URL only (e.g., http://10.140.141.160:48888/) - do NOT include /tree/ path",
            key="jupyter_url_input",
        )
        jupyter_config["server_url"] = jupyter_url

        jupyter_token = st.text_input(
            "Jupyter Token:",
            value=jupyter_config.get("token", ""),
            type="password",
            help="Authentication token for Jupyter server",
            key="jupyter_token_input",
        )
        jupyter_config["token"] = jupyter_token

        jupyter_notebook_path = st.text_input(
            "Base Notebook Path/Directory:",
            value=jupyter_config.get("notebook_path", "Automated Agent"),
            help="Base directory path in Jupyter (e.g., 'Automated Agent'). Curve fitting will create subfolders with filename_date",
            key="jupyter_notebook_path_input",
        )
        jupyter_config["notebook_path"] = jupyter_notebook_path

        jupyter_upload_enabled = st.checkbox(
            "Enable Auto-Upload to Jupyter",
            value=jupyter_config.get("upload_enabled", False),
            help="Automatically upload generated files to Jupyter server (applies to all agents)",
            key="jupyter_upload_enabled_input",
        )
        jupyter_config["upload_enabled"] = jupyter_upload_enabled
        memory.set_var("jupyter_config", jupyter_config)

    with col_exp2:
        st.markdown("**Experiment Memory**")
        experiment_memory_file = st.text_input(
            "Experiment Memory File:",
            value=memory.get_var("experiment_memory_file", "experiment_memory.json"),
            help="File to store completed experiment records",
            key="exp_memory_file_input",
        )
        memory.set_var("experiment_memory_file", experiment_memory_file)

        experiment_data_dir = st.text_input(
            "Experiment Data Directory:",
            value=memory.get_var("experiment_data_dir", "data"),
            help="Directory where experiment data and memory files are stored",
            key="exp_data_dir_input",
        )
        memory.set_var("experiment_data_dir", experiment_data_dir)

with cache:
    st.markdown("##### Cache Management")
    st.markdown("Clear cached data from the application to force fresh computations.")

    col_cache1, col_cache2 = st.columns(2)

    with col_cache1:
        st.markdown("**Streamlit Cache**")
        st.markdown("Clear Streamlit's built-in cache decorators (`@st.cache_data`, `@st.cache_resource`).")

        if st.button("Clear Streamlit Cache", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Streamlit cache cleared successfully!")
            st.rerun()

    with col_cache2:
        st.markdown("**Session State**")
        st.markdown("Reset session state variables (this will restart your session).")

        if st.button("Clear Session State", type="secondary", use_container_width=True):
            memory.clear_session_state(keep_keys=["start_time", "api_key", "api_key_source"])
            st.success("Session state cleared! Page will reload.")
            st.rerun()

    st.markdown("---")
    st.markdown("**Clear Everything**")
    st.markdown("**Warning:** This will clear all caches and reset your session completely.")

    if st.button("Clear All Caches & Reset Session", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        memory.clear_all()
        st.success("All caches cleared and session reset!")
        st.rerun()

    st.markdown("---")
    st.markdown("**Cache Statistics**")

    try:
        cache_info = st.cache_data.get_stats()
        if cache_info:
            st.json(cache_info)
        else:
            st.info("No cache statistics available.")
    except Exception:
        st.info("Cache statistics not available in this Streamlit version.")
