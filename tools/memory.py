"""
MemoryManager - Unified storage for Polaris Ahmadi.
Uses SQLite database for persistence and runtime_state for non-serializable objects.
Replaces Streamlit session state entirely.
"""

from datetime import datetime
import json
import logging
import os
import uuid

# Lazy import streamlit to avoid issues in headless mode
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except (ImportError, RuntimeError):
    STREAMLIT_AVAILABLE = False
    st = None

from tools.database import DatabaseManager
from tools.paths import get_env_path, get_user_data_dir
from tools.runtime_state import clear_ephemeral, get as runtime_get, is_runtime_key, set as runtime_set


class MemoryManager:
    def __init__(self):
        self._db = DatabaseManager()

    def init_session(self):
        """Initialize the session: ensure DB defaults and handle API key from env/secrets."""
        self._db.ensure_defaults()
        clear_ephemeral()

        # Set start_time if not present
        if self._db.get("start_time") is None:
            self._db.set("start_time", datetime.now().timestamp())

        # Load .env from user data dir when frozen (PyInstaller)
        try:
            from tools.paths import is_frozen
            if is_frozen():
                env_path = get_env_path()
                if os.path.exists(env_path):
                    from dotenv import load_dotenv
                    load_dotenv(env_path)
        except ImportError:
            pass

        # API key: only override if user hasn't manually set one
        if not STREAMLIT_AVAILABLE:
            return

        api_key_source = self._db.get("api_key_source", "")
        if api_key_source != "user":
            env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

            if env_key:
                self._db.set("api_key", env_key)
                self._db.set("api_key_source", "environment")
                os.environ["GOOGLE_API_KEY"] = env_key
                os.environ["GEMINI_API_KEY"] = env_key

            elif not self._db.get("api_key"):
                try:
                    api_key = st.secrets.get("GEMINI_API_KEY")
                    if api_key:
                        self._db.set("api_key", api_key)
                        self._db.set("api_key_source", "secrets")
                        os.environ["GEMINI_API_KEY"] = api_key
                        os.environ["GOOGLE_API_KEY"] = api_key
                except (AttributeError, RuntimeError):
                    pass
        else:
            api_key = self._db.get("api_key")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
                os.environ["GEMINI_API_KEY"] = api_key

    def log_event(self, event_type: str, payload: dict, mode: str):
        """Unified event log to database (or logger if DB unavailable)."""
        try:
            prompt_session_id = self._db.get("current_prompt_session_id", "")
            if not prompt_session_id:
                prompt_session_id = str(uuid.uuid4())
                self._db.set("current_prompt_session_id", prompt_session_id)
            self._db.append_conversation_event(
                event_type, mode, prompt_session_id, payload,
            )
            # Also append to interactions list for UI (stored in session_state)
            if event_type == "interaction":
                interactions = self._db.get("interactions", [])
                interactions.append({
                    "role": payload.get("role", ""),
                    "message": payload.get("message", ""),
                    "component": payload.get("component", ""),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                self._db.set("interactions", interactions)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.info(f"Event [{mode}]: {event_type} - {payload} ({e})")

    def save_to_history(self, question, mode, hypothesis=None, thoughts=None):
        """Log important parts of hypothesis agent conversation."""
        self.log_event(
            "history",
            {"question": question, "hypothesis": hypothesis, "thoughts": thoughts},
            mode=mode,
        )

    def insert_interaction(self, role, message, component, mode):
        """Add interactions for history tracking and UI display."""
        self.log_event(
            "interaction",
            {"role": role, "message": message, "component": component},
            mode=mode,
        )

    def get_latest_history(self, mode=None):
        """Get latest conversation history."""
        events = self._db.get_conversation_events()
        for event in reversed(events):
            if event["type"] == "history":
                if mode is None or event.get("mode") == mode:
                    return event
        return None

    def get_prompt_session_events(self, prompt_session_id: str):
        """Get all events from prompt session."""
        events = self._db.get_conversation_events()
        return [e for e in events if e.get("prompt_session_id") == prompt_session_id]

    def get_prompt_session_history(self, prompt_session_id: str):
        """Get all history from prompt session."""
        events = self._db.get_conversation_events()
        return [
            e for e in events
            if e["type"] == "history" and e.get("prompt_session_id") == prompt_session_id
        ]

    def get_prompt_session_interactions(self, prompt_session_id: str):
        """Get latest interactions from prompt session."""
        events = self._db.get_conversation_events()
        return [
            e for e in events
            if e["type"] == "interaction" and e.get("prompt_session_id") == prompt_session_id
        ]

    def view_component(self, component, prompt_session_id: str = None):
        """Get component value from conversation events."""
        if prompt_session_id is None:
            prompt_session_id = self._db.get("current_prompt_session_id", "")
        events = self._db.get_conversation_events()
        for event in reversed(events):
            if event["type"] == "interaction" and event.get("prompt_session_id") == prompt_session_id:
                payload = event.get("payload", {})
                if payload.get("component") == component:
                    return payload.get("message")
        return None

    def add_uploaded_file(self, filename, path):
        """Add uploaded file metadata to database."""
        self._db.add_uploaded_file(filename, path)

    def save_workflow(self, workflow_name: str, steps: list, ml_model_choice=None):
        """Save workflow to database."""
        self._db.save_workflow(workflow_name, steps, ml_model_choice)

    def load_workflow(self, workflow_name: str):
        """Load workflow from database."""
        workflows = self._db.get_workflows()
        return workflows.get(workflow_name)

    def delete_workflow(self, workflow_name: str):
        """Delete workflow from database."""
        self._db.delete_workflow(workflow_name)

    def snapshot_session_state(self, note: str = "session_state_snapshot"):
        """Capture a JSON-serializable snapshot of current state for debugging."""
        serialisable = {}
        for key in [
            "stage", "workflow_active", "workflow_step", "hypothesis_ready",
            "research_goal", "api_key_source", "routing_mode",
        ]:
            val = self.get_var(key)
            try:
                json.dumps(val)
                serialisable[key] = val
            except (TypeError, ValueError):
                pass
        self.log_event("session_state", {"note": note, "state": serialisable}, mode="system")

    def get_var(self, name: str, default=None):
        """Safe accessor: routes to DB or runtime based on key."""
        if is_runtime_key(name):
            return runtime_get(name, default)
        return self._db.get(name, default)

    def set_var(self, name: str, value):
        """Safe setter: routes to DB or runtime based on key."""
        if is_runtime_key(name):
            runtime_set(name, value)
            return
        self._db.set(name, value)

    def delete_var(self, name: str):
        """Delete a variable (mainly for runtime keys like process handles)."""
        if is_runtime_key(name):
            from tools.runtime_state import delete
            delete(name)
            return
        # For DB keys, set to None or clear from session_state
        self._db.set(name, None)

    def clear_session_state(self, keep_keys=None):
        """Clear session state except for keep_keys."""
        keep_keys = keep_keys or ["start_time", "api_key", "api_key_source"]
        self._db.clear_session_state(keep_keys)
        clear_ephemeral()

    def clear_all(self):
        """Clear all caches and reset session."""
        self._db.clear_all_except(["start_time"])
        clear_ephemeral()
        self.init_session()
