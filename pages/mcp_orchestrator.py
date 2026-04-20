"""
MCP Orchestrator Control Page - start/stop and test orchestrator server.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="centered")
st.title("MCP Orchestrator")
st.markdown("Start/stop the orchestrator server and validate MCP tool connectivity.")

mcp_cfg = memory.get_var("mcp_literature_config", {}) or {}
default_endpoint = mcp_cfg.get("endpoint", "http://127.0.0.1:8000/mcp")
default_manifest = mcp_cfg.get("manual_manifest_path", "data/manual_papers_manifest.json")
default_host = memory.get_var("mcp_orch_host", "127.0.0.1")
default_port = int(memory.get_var("mcp_orch_port", 8010) or 8010)

tabs = st.tabs(["Configuration", "Server", "Tools", "Navigation"])


def orchestrator_url() -> str:
    host = memory.get_var("mcp_orch_host", "127.0.0.1")
    port = int(memory.get_var("mcp_orch_port", 8010) or 8010)
    return f"http://{host}:{port}"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def start_orchestrator_server() -> tuple[bool, str]:
    try:
        base_url = orchestrator_url()
        try:
            resp = requests.get(f"{base_url}/health", timeout=1.2)
            if resp.status_code == 200:
                return True, "Orchestrator server is already running."
        except Exception:
            pass

        project_root = _project_root()
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"mcp_orchestrator_{timestamp}.log"
        log_handle = open(log_file, "w", encoding="utf-8")

        env = os.environ.copy()
        env["MCP_ORCH_HOST"] = memory.get_var("mcp_orch_host", "127.0.0.1")
        env["MCP_ORCH_PORT"] = str(int(memory.get_var("mcp_orch_port", 8010) or 8010))
        env["LITERATURE_MCP_ENDPOINT"] = memory.get_var("mcp_literature_config", {}).get(
            "endpoint", "http://127.0.0.1:8000/mcp"
        )
        env["MANUAL_PAPER_MANIFEST"] = memory.get_var("mcp_literature_config", {}).get(
            "manual_manifest_path", "data/manual_papers_manifest.json"
        )

        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        process = subprocess.Popen(
            [sys.executable, "watcher/orchestrator_mcp.py"],
            cwd=str(project_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
        )

        memory.set_var("mcp_orchestrator_process", process)
        memory.set_var("mcp_orchestrator_pid", process.pid)
        memory.set_var("mcp_orchestrator_log_file", str(log_file))
        memory.set_var("mcp_orchestrator_log_file_handle", log_handle)

        time.sleep(1.5)
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                return True, f"Orchestrator started successfully (PID: {process.pid})"
        except Exception:
            pass
        return True, "Process started. Refresh status if health endpoint is not ready yet."
    except Exception as exc:
        return False, f"Failed to start orchestrator: {exc}"


def stop_orchestrator_server() -> tuple[bool, str]:
    try:
        process = memory.get_var("mcp_orchestrator_process")
        if process is None:
            return False, "No orchestrator process tracked in this session."

        try:
            if sys.platform == "win32":
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

        log_handle = memory.get_var("mcp_orchestrator_log_file_handle")
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
            memory.delete_var("mcp_orchestrator_log_file_handle")

        memory.delete_var("mcp_orchestrator_process")
        memory.delete_var("mcp_orchestrator_pid")
        return True, "Orchestrator server stopped."
    except Exception as exc:
        return False, f"Failed to stop orchestrator: {exc}"


with tabs[0]:
    st.markdown("##### Orchestrator Configuration")
    orch_host = st.text_input("Orchestrator Host", value=default_host, key="mcp_orch_host_input")
    orch_port = st.number_input("Orchestrator Port", min_value=1024, max_value=65535, value=default_port, key="mcp_orch_port_input")
    mcp_endpoint = st.text_input("Literature MCP Endpoint", value=default_endpoint, key="mcp_lit_endpoint_input")
    manual_manifest = st.text_input("Manual Paper Manifest Path", value=default_manifest, key="mcp_manual_manifest_input")

    memory.set_var("mcp_orch_host", orch_host)
    memory.set_var("mcp_orch_port", int(orch_port))
    memory.set_var(
        "mcp_literature_config",
        {"endpoint": mcp_endpoint, "manual_manifest_path": manual_manifest},
    )

    st.info(f"Orchestrator URL: `{orchestrator_url()}`")

with tabs[1]:
    st.markdown("##### Server Control")
    base_url = orchestrator_url()
    server_running = False
    try:
        resp = requests.get(f"{base_url}/health", timeout=2)
        if resp.status_code == 200:
            server_running = True
            st.success(f"Server running at `{base_url}`")
            st.json(resp.json())
    except Exception:
        st.info("Server not running.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start Server", use_container_width=True, type="primary"):
            ok, msg = start_orchestrator_server()
            st.success(msg) if ok else st.error(msg)
            st.rerun()
    with col2:
        if st.button("Stop Server", use_container_width=True):
            ok, msg = stop_orchestrator_server()
            st.success(msg) if ok else st.warning(msg)
            st.rerun()
    with col3:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    log_path = memory.get_var("mcp_orchestrator_log_file", "")
    if log_path:
        st.caption(f"Log file: `{log_path}`")

with tabs[2]:
    st.markdown("##### Tool Access Checks")
    base_url = orchestrator_url()

    if st.button("List MCP Tools", use_container_width=True):
        try:
            resp = requests.get(f"{base_url}/tools", timeout=10)
            st.write(f"Status: {resp.status_code}")
            if "application/json" in resp.headers.get("content-type", ""):
                st.json(resp.json())
            else:
                st.code(resp.text)
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    search_query = st.text_input("Search Query", value="perovskite solar cell stability")
    source_mode = st.selectbox("Source Mode", ["hybrid", "manual_only", "mcp_only"], index=0)
    if st.button("Search Papers", use_container_width=True):
        try:
            payload = {
                "query": search_query,
                "year_min": 2021,
                "year_max": 2026,
                "max_candidates": 5,
                "source_mode": source_mode,
            }
            resp = requests.post(f"{base_url}/search-papers", json=payload, timeout=30)
            st.write(f"Status: {resp.status_code}")
            if "application/json" in resp.headers.get("content-type", ""):
                st.json(resp.json())
            else:
                st.code(resp.text)
        except Exception as exc:
            st.error(f"Request failed: {exc}")

    st.divider()
    st.markdown("##### Process Tools")

    process_action = st.selectbox(
        "Process action",
        [
            "list_processed_papers",
            "get_saved_paper_output",
            "process_batch",
        ],
        index=0,
        key="mcp_orch_process_action",
    )

    paper_slug = st.text_input(
        "Paper slug (for get_saved_paper_output)",
        value="",
        key="mcp_orch_paper_slug",
    )
    batch_query = st.text_input(
        "Batch query (for process_batch)",
        value=search_query,
        key="mcp_orch_batch_query",
    )
    batch_max = st.number_input(
        "Batch max papers",
        min_value=1,
        max_value=20,
        value=1,
        key="mcp_orch_batch_max",
    )
    batch_run_mode = st.selectbox(
        "Batch run mode",
        ["resume", "expand", "reprocess", "reset"],
        index=0,
        key="mcp_orch_batch_run_mode",
    )
    batch_force_reprocess = st.checkbox(
        "Force reprocess",
        value=False,
        key="mcp_orch_force_reprocess",
    )
    batch_reset_output = st.checkbox(
        "Reset output",
        value=False,
        key="mcp_orch_reset_output",
    )

    if st.button("Run Process Action", use_container_width=True, key="mcp_orch_run_process"):
        try:
            payload: dict = {"action": process_action}
            if process_action == "get_saved_paper_output":
                if not paper_slug.strip():
                    st.warning("Please enter paper slug for get_saved_paper_output.")
                    st.stop()
                payload["paper_slug"] = paper_slug.strip()
            elif process_action == "process_batch":
                payload.update(
                    {
                        "query": batch_query,
                        "year_min": 2021,
                        "year_max": 2026,
                        "max_papers": int(batch_max),
                        "run_mode": batch_run_mode,
                        "force_reprocess": bool(batch_force_reprocess),
                        "reset_output": bool(batch_reset_output),
                    }
                )

            resp = requests.post(f"{base_url}/process-paper", json=payload, timeout=120)
            st.write(f"Status: {resp.status_code}")
            if "application/json" in resp.headers.get("content-type", ""):
                body = resp.json()
                st.json(body)
                if process_action == "list_processed_papers":
                    result = body.get("result", {}) if isinstance(body, dict) else {}
                    papers = []
                    if isinstance(result, dict):
                        papers = result.get("papers") or result.get("processed_papers") or []
                    if papers and isinstance(papers, list):
                        first = papers[0]
                        if isinstance(first, dict):
                            auto_slug = first.get("paper_slug") or first.get("slug")
                            if auto_slug:
                                st.info(f"Suggested slug for saved output: `{auto_slug}`")
            else:
                st.code(resp.text)
        except Exception as exc:
            st.error(f"Request failed: {exc}")

with tabs[3]:
    st.markdown("##### Quick Navigation")
    st.page_link("pages/watcher_control.py", label="Open Watcher Control")
    st.page_link("pages/settings.py", label="Open Settings")
    st.page_link("pages/history.py", label="Open History")
    st.page_link("pages/hypothesis.py", label="Open Hypothesis Agent")
    st.page_link("pages/analysis.py", label="Open Analysis Agent")
