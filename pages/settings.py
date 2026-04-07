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

    # LLM Provider and Model
    st.markdown("##### LLM Provider")
    llm_provider = st.selectbox(
        "Model provider:",
        ["gemini", "qwen"],
        format_func=lambda x: "Google Gemini" if x == "gemini" else "Qwen 2.5 (Alibaba DashScope)",
        index=0 if (memory.get_var("llm_provider") or "gemini") == "gemini" else 1,
        key="llm_provider_select",
    )
    memory.set_var("llm_provider", llm_provider)

    if llm_provider == "gemini":
        llm_model = st.text_input(
            "Gemini model ID:",
            value=memory.get_var("llm_model") or "gemini-2.5-flash-lite",
            help="e.g. gemini-2.5-flash-lite, gemini-2.0-flash-lite",
            key="llm_model_gemini",
        )
    else:
        llm_model = st.selectbox(
            "Qwen model:",
            ["qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct", "qwen-plus", "qwen-turbo"],
            index=0,
            key="llm_model_qwen",
        )
        custom_model = st.text_input("Or custom model ID:", value="", key="qwen_custom_model", placeholder="e.g. qwen3-32b")
        if custom_model.strip():
            llm_model = custom_model.strip()
        qwen_base_url = st.selectbox(
            "DashScope region:",
            [
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            ],
            format_func=lambda u: "Beijing" if "us" not in u and "intl" not in u else ("International" if "intl" in u else "US"),
            key="qwen_base_url",
        )
        memory.set_var("qwen_base_url", qwen_base_url)
    memory.set_var("llm_model", llm_model)

    # API Key
    st.markdown("##### API Key Configuration")
    key_label = "DashScope API Key (Qwen):" if llm_provider == "qwen" else "Google Gemini API Key:"
    key_help = "Get key from https://dashscope.console.aliyun.com/" if llm_provider == "qwen" else "Get key from https://makersuite.google.com/app/apikey"

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
            key_label,
            value="" if editing else memory.get_var("api_key", ""),
            type="password",
            help=key_help,
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
                    os.environ["LLM_API_KEY"] = api_key
                    os.environ["LLM_PROVIDER"] = llm_provider
                    os.environ["LLM_MODEL"] = llm_model
                    if llm_provider == "gemini":
                        os.environ["GEMINI_API_KEY"] = api_key
                        os.environ["GOOGLE_API_KEY"] = api_key
                    else:
                        os.environ["DASHSCOPE_API_KEY"] = api_key
                        os.environ["QWEN_BASE_URL"] = memory.get_var("qwen_base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    try:
                        env_path = get_env_path()
                        with open(env_path, "w") as f:
                            f.write(f"LLM_API_KEY={api_key}\nLLM_PROVIDER={llm_provider}\nLLM_MODEL={llm_model}\n")
                            if llm_provider == "qwen":
                                f.write(f"QWEN_BASE_URL={memory.get_var('qwen_base_url') or 'https://dashscope.aliyuncs.com/compatible-mode/v1'}\n")
                    except Exception:
                        pass
                    st.success("Your API key and settings have been saved successfully!")
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
    st.markdown("##### Record Negative Hypothesis")
    st.markdown("Manually add hypotheses that did not work (e.g., from lab notebooks). The model uses these to avoid similar mistakes.")
    with st.expander("Add past negative hypothesis", expanded=False):
        manual_hypothesis = st.text_area(
            "Hypothesis text:",
            placeholder="Paste the hypothesis that did not work...",
            key="manual_neg_hypothesis",
        )
        manual_status = st.selectbox(
            "Status:",
            ["rejected", "needs_revision"],
            format_func=lambda x: "Rejected" if x == "rejected" else "Needs revision",
            key="manual_neg_status",
        )
        manual_reason = st.text_area(
            "Why it didn't work:",
            placeholder="e.g., Experimental data contradicted predictions; R² was too low...",
            key="manual_neg_reason",
        )
        manual_research_q = st.text_input(
            "Research question (optional):",
            placeholder="The clarified question this hypothesis addressed",
            key="manual_neg_research_q",
        )
        if st.button("Save negative hypothesis", use_container_width=True, key="save_manual_neg"):
            if manual_hypothesis and manual_hypothesis.strip():
                memory.add_negative_hypothesis(
                    hypothesis_text=manual_hypothesis[:4000],
                    status=manual_status,
                    research_question=manual_research_q or "",
                    analysis_summary=manual_reason[:2000] if manual_reason else "",
                )
                st.success("Saved. The model will use this when generating new hypotheses.")
                st.rerun()
            else:
                st.error("Please enter the hypothesis text.")

    # Show stored negative hypotheses
    neg_hyps = memory.get_negative_hypotheses(limit=10)
    if neg_hyps:
        with st.expander(f"View stored negative hypotheses ({len(neg_hyps)} recent)", expanded=False):
            for i, nh in enumerate(neg_hyps, 1):
                st.markdown(f"**{i}. [{nh.get('status', '?')}]** {nh.get('created_at', '')}")
                st.caption((nh.get("hypothesis_text") or "")[:200] + ("..." if len(nh.get("hypothesis_text", "") or "") > 200 else ""))
    else:
        st.caption("No negative hypotheses recorded yet.")

    st.markdown("---")
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
