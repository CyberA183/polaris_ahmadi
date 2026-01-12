import streamlit as st
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="centered")
st.title("Settings")
st.markdown("Adjust the settings for the various agents and workflows below.")

general, experiment, watcher, cache = st.tabs(
    ["General", "Experiment", "Watcher", "Cache"]
)

with general:

    # Changing API Key
    st.markdown("##### API Key Configuration")

    # Track editing mode
    if "editing" not in st.session_state:
        st.session_state.editing = False

    if st.session_state.api_key and not st.session_state.editing:
        st.success("Your API key was loaded from session state successfully.")

        if st.button("Edit API Key"):
            st.session_state.editing = True
            # Clear the input field when entering edit mode
            if "api_key_input" in st.session_state:
                del st.session_state.api_key_input
            st.rerun()

    else:
        # Show current status
        if st.session_state.get('api_key_source') == 'environment':
            st.info("📝 Currently using API key from environment variables. Click 'Edit API Key' to set a custom one.")
        elif st.session_state.get('api_key_source') == 'secrets':
            st.info("📝 Currently using API key from Streamlit secrets. Click 'Edit API Key' to set a custom one.")

        api_key_input = st.text_input(
            "Google Gemini API Key:",
            value="" if st.session_state.editing else st.session_state.get('api_key', ''),
            type="password",
            help="Enter your Google Gemini API key. It will be saved securely and persist across page navigations.",
            key="api_key_input",
            placeholder="Enter your API key here..." if st.session_state.editing else None,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Save API Key", use_container_width=True):
                if api_key_input and api_key_input.strip():
                    api_key = api_key_input.strip()
                    st.session_state.api_key = api_key
                    st.session_state.api_key_source = "user"
                    st.session_state.editing = False
                    # Also save to environment variable for persistence across page navigations
                    os.environ['GEMINI_API_KEY'] = api_key
                    # Try to save to .env file for persistence across app restarts
                    try:
                        env_path = Path(__file__).parent.parent / '.env'
                        with open(env_path, 'w') as f:
                            f.write(f'GEMINI_API_KEY={api_key}\n')
                    except Exception:
                        pass  # Ignore if we can't write to .env file
                    st.success("Your API key has been saved successfully!")
                    st.rerun()
                else:
                    st.error("Please enter your API key and try again.")

        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.editing = False
                st.rerun()

    st.markdown("---")
    st.markdown("##### Workflow & Routing")

    # Routing mode control
    routing_mode = st.segmented_control(
        "Routing Mode",
        options=["Autonomous (LLM)", "Manual"],
        default=st.session_state.get("routing_mode", "Autonomous (LLM)"),
    )
    st.session_state.routing_mode = routing_mode

    # Manual workflow configuration
    if routing_mode == "Manual":
        st.info(
            "In manual mode, agents run in the order you specify below whenever the router is invoked."
        )

        available_agents = ["Hypothesis Agent", "Experiment Agent", "Curve Fitting"]
        current_workflow = st.session_state.get(
            "manual_workflow", ["Hypothesis Agent", "Experiment Agent", "Curve Fitting"]
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            step1 = st.selectbox(
                "Step 1",
                options=available_agents,
                index=available_agents.index(current_workflow[0])
                if current_workflow and current_workflow[0] in available_agents
                else 0,
            )
        with col_b:
            step2 = st.selectbox(
                "Step 2",
                options=available_agents,
                index=available_agents.index(current_workflow[1])
                if len(current_workflow) > 1 and current_workflow[1] in available_agents
                else 1,
            )
        with col_c:
            step3 = st.selectbox(
                "Step 3",
                options=available_agents,
                index=available_agents.index(current_workflow[2])
                if len(current_workflow) > 2 and current_workflow[2] in available_agents
                else 2,
            )

        new_workflow = [step1, step2, step3]
        st.session_state.manual_workflow = new_workflow

        if st.button("Reset Workflow Progress"):
            st.session_state.workflow_index = 0
            st.success("Workflow progress reset. The next routed call will start at Step 1.")

with experiment:
    # Changing Jupyter Sever Configuration
    st.markdown("##### Jupyter Server Configuration")

    with st.form("jupyter_server_config"):
        col_jup1, col_jup2 = st.columns(2)

        with col_jup1:
            jupyter_url = st.text_input(
                "Jupyter Server URL:",
                value=st.session_state.jupyter_config["server_url"],
                help="URL of Jupyter server (e.g., http://10.140.141.160:48888/)",
            )

            jupyter_token = st.text_input(
                "Jupyter Token (optional):",
                value=st.session_state.jupyter_config["token"],
                type="password",
                help="Authentication token for Jupyter server",
            )

        with col_jup2:
            jupyter_notebook_path = st.text_input(
                "Notebook Path:",
                value=st.session_state.jupyter_config["notebook_path"],
                help="Directory path in Jupyter (e.g., 'Dual GP 5AVA BDA')",
            )

            jupyter_upload_enabled = st.checkbox(
                "Enable Auto-Upload to Jupyter",
                value=st.session_state.jupyter_config["upload_enabled"],
                help="Automatically upload generated files to Jupyter server",
            )

        submitted = st.form_submit_button(
            "Update Jupyter Server Configuration", use_container_width=True
        )

    if submitted:
        st.session_state.jupyter_config = {
            "server_url": jupyter_url,
            "token": jupyter_token,
            "upload_enabled": jupyter_upload_enabled,
            "notebook_path": jupyter_notebook_path,
        }

        st.success("Jupyter Server Configuration loaded successfully!")

with watcher:
    st.markdown("##### Watcher Agent Configuration")
    
    st.markdown("""
    The Watcher Agent monitors filesystem events and automatically triggers the next agent in your workflow.
    It watches for files matching the pattern `output_from_*.json` and routes them to the appropriate agent.
    """)
    
    # Server Configuration
    st.markdown("**Server Configuration**")
    watcher_config_col1, watcher_config_col2 = st.columns(2)
    
    with watcher_config_col1:
        watcher_server_url = st.text_input(
            "Watcher Server URL:",
            value=st.session_state.get("watcher_server_url", "http://localhost:8000"),
            help="URL of the FastAPI watcher server (default: http://localhost:8000)",
            key="watcher_server_url_input"
        )
        if st.button("Save Watcher URL", use_container_width=True):
            st.session_state.watcher_server_url = watcher_server_url
            st.success("Watcher server URL saved!")
    
    with watcher_config_col2:
        watch_directory = st.text_input(
            "Watch Directory:",
            value=st.session_state.get("watcher_watch_dir", str(Path.cwd())),
            help="Directory to watch for filesystem events (default: current working directory)",
            key="watcher_watch_dir_input"
        )
        if st.button("Save Watch Directory", use_container_width=True):
            st.session_state.watcher_watch_dir = watch_directory
            st.success("Watch directory saved!")
    
    st.markdown("---")
    
    # Server Control
    st.markdown("**Server Control**")
    
    # Check server status
    server_url = st.session_state.get("watcher_server_url", "http://localhost:8000")
    server_running = False
    server_info = {}
    
    try:
        import requests
        response = requests.get(f"{server_url}/health", timeout=2)
        if response.status_code == 200:
            server_info = response.json()
            server_running = server_info.get("observer_running", False)
    except Exception:
        server_running = False
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if server_running:
            st.success("✅ **Server Running**")
            if server_info.get("watch_dir"):
                st.caption(f"Watching: {server_info.get('watch_dir')}")
        else:
            st.error("❌ **Server Not Running**")
            st.caption("Start the server to begin watching")
    
    with status_col2:
        if not server_running:
            if st.button("▶️ Start Watcher Server", use_container_width=True, type="primary"):
                try:
                    # Get the project root directory
                    project_root = Path(__file__).parent.parent
                    server_script = project_root / "watcher" / "server.py"
                    
                    if not server_script.exists():
                        st.error(f"Watcher server script not found at: {server_script}")
                    else:
                        # Start server in background
                        # Use subprocess to start the server
                        env = os.environ.copy()
                        if st.session_state.get("watcher_watch_dir"):
                            env["WATCH_DIR"] = st.session_state.watcher_watch_dir
                        if st.session_state.get("watcher_server_url"):
                            # Extract port from URL
                            try:
                                from urllib.parse import urlparse
                                parsed = urlparse(st.session_state.watcher_server_url)
                                if parsed.port:
                                    env["WATCHER_PORT"] = str(parsed.port)
                            except:
                                pass
                        
                        # Start the server process
                        process = subprocess.Popen(
                            [sys.executable, str(server_script)],
                            cwd=str(project_root),
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            start_new_session=True
                        )
                        
                        # Store process info
                        st.session_state.watcher_server_pid = process.pid
                        st.session_state.watcher_server_process = process
                        
                        st.success("🚀 Watcher server starting...")
                        st.info("💡 The server is running in the background. Check status above to confirm it's running.")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to start server: {str(e)}")
                    st.info("💡 You can also start it manually: `python watcher/server.py`")
        else:
            if st.button("⏹️ Stop Watcher Server", use_container_width=True, type="secondary"):
                try:
                    import requests
                    response = requests.post(f"{server_url}/watch/stop", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Server stopped successfully!")
                        if "watcher_server_pid" in st.session_state:
                            del st.session_state.watcher_server_pid
                        if "watcher_server_process" in st.session_state:
                            del st.session_state.watcher_server_process
                        st.rerun()
                    else:
                        st.warning("⚠️ Server stop request sent, but response was unexpected")
                except Exception as e:
                    st.error(f"❌ Failed to stop server: {str(e)}")
                    st.info("💡 You may need to stop it manually or restart the Streamlit app")
    
    with status_col3:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # Manual Trigger
    st.markdown("**Manual Watcher Trigger**")
    st.markdown("Manually trigger the watcher agent to process a file event:")
    
    col_trigger1, col_trigger2 = st.columns(2)
    
    with col_trigger1:
        trigger_file_path = st.text_input(
            "File Path:",
            value="output_from_hypothesis.json",
            help="Path to the file that should trigger the watcher",
            key="trigger_file_path"
        )
    
    with col_trigger2:
        if st.button("🚀 Trigger Watcher", use_container_width=True, type="primary"):
            if not server_running:
                st.warning("⚠️ Server is not running. Please start the server first.")
            else:
                try:
                    import requests
                    server_url = st.session_state.get("watcher_server_url", "http://localhost:8000")
                    
                    # Create file event request
                    event_data = {
                        "file_path": trigger_file_path,
                        "event_type": "created",
                        "metadata": {
                            "triggered_by": "user",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    response = requests.post(
                        f"{server_url}/file-event",
                        json=event_data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ Watcher agent triggered successfully!")
                        result = response.json()
                        with st.expander("View Response", expanded=False):
                            st.json(result)
                    else:
                        st.error(f"❌ Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"❌ Failed to trigger watcher: {str(e)}")
    
    st.markdown("---")
    
    # Instructions
    with st.expander("📖 Manual Start Instructions", expanded=False):
        st.markdown("""
        **To start the watcher server manually:**
        
        1. Open a terminal/command prompt
        2. Navigate to the project directory
        3. Run one of the following commands:
        
        ```bash
        # Using Python directly
        python watcher/server.py
        
        # Using uvicorn
        uvicorn watcher.server:app --host 0.0.0.0 --port 8000
        
        # With custom port
        WATCHER_PORT=8001 python watcher/server.py
        
        # With custom watch directory
        WATCH_DIR=/path/to/watch python watcher/server.py
        ```
        
        **Server Endpoints:**
        - `GET /health` - Check server status
        - `POST /watch/start?watch_directory=/path` - Start watching a directory
        - `POST /watch/stop` - Stop watching
        - `POST /file-event` - Manually trigger a file event
        - `POST /route` - Route a payload directly
        """)

with cache:
    st.markdown("##### Cache Management")
    st.markdown("Clear cached data from the Streamlit application to force fresh computations.")
    
    col_cache1, col_cache2 = st.columns(2)
    
    with col_cache1:
        st.markdown("**Streamlit Cache**")
        st.markdown("Clear Streamlit's built-in cache decorators (`@st.cache_data`, `@st.cache_resource`).")
        
        if st.button("Clear Streamlit Cache", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Streamlit cache cleared successfully!")
            st.rerun()
    
    with col_cache2:
        st.markdown("**Session State**")
        st.markdown("Reset session state variables (this will restart your session).")
        
        if st.button("Clear Session State", type="secondary", use_container_width=True):
            # Clear all session state except essential keys
            essential_keys = ["start_time", "api_key", "api_key_source"]
            keys_to_clear = [k for k in st.session_state.keys() if k not in essential_keys]
            for key in keys_to_clear:
                del st.session_state[key]
            st.success("✅ Session state cleared! Page will reload.")
            st.rerun()
    
    st.markdown("---")
    st.markdown("**Clear Everything**")
    st.markdown("⚠️ **Warning:** This will clear all caches and reset your session completely.")
    
    if st.button("Clear All Caches & Reset Session", type="primary", use_container_width=True):
        # Clear Streamlit caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Clear session state (keep only minimal essentials)
        essential_keys = ["start_time"]
        keys_to_clear = [k for k in st.session_state.keys() if k not in essential_keys]
        for key in keys_to_clear:
            del st.session_state[key]
        
        # Reinitialize session
        memory.init_session()
        
        st.success("✅ All caches cleared and session reset!")
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Cache Statistics**")
    
    # Display cache info if available
    try:
        cache_info = st.cache_data.get_stats()
        if cache_info:
            st.json(cache_info)
        else:
            st.info("No cache statistics available.")
    except Exception:
        st.info("Cache statistics not available in this Streamlit version.")