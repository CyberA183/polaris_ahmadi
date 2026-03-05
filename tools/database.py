"""
3NF SQLite database for Polaris Ahmadi.
Replaces Streamlit session state with persistent storage.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.paths import get_db_path

# Thread-local connection for Streamlit's multi-threaded model
_local = threading.local()

# Default values for app config (from memory.py)
DEFAULT_APP_CONFIG = {
    "start_time": None,  # Set at init
    "api_key": "",
    "api_key_source": "",
    "stage": "initial",
    "cf_data_path": "",
    "cf_comp_path": "",
    "experimental_mode": False,
    "routing_mode": "Autonomous (LLM)",
    "max_hypothesis_rounds": 5,
    "experiment_memory_file": "experiment_memory.json",
    "experiment_data_dir": "data",
    "watcher_directory": "",
    "watcher_results_dir": "results",
    "watcher_enabled": False,
    "watcher_port": 8000,
}

DEFAULT_EXPERIMENTAL_CONSTRAINTS = {
    "techniques": [],
    "equipment": [],
    "parameters": [],
    "focus_areas": [],
    "liquid_handling": {
        "max_volume_per_mixture": 50,
        "instruments": [],
        "plate_format": "96-well",
        "materials": [],
        "csv_path": "/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/",
    },
}

DEFAULT_JUPYTER_CONFIG = {
    "server_url": "http://10.140.141.160:48888/",
    "token": "",
    "upload_enabled": False,
    "notebook_path": "Automated Agent",
}

DEFAULT_SESSION_STATE = {
    "conversation_events": [],
    "current_prompt_session_id": None,
    "manual_workflow": ["Hypothesis Agent", "Experiment Agent", "Curve Fitting"],
    "workflow_index": 0,
    "workflow_auto_flags": {},
    "current_workflow_name": None,
    "uploaded_files": [],
    "hypothesis_ready": False,
    "last_hypothesis": None,
    "experimental_outputs": None,
    "stop_hypothesis": False,
    "hypothesis_round_count": 0,
    "workflow_active": False,
    "workflow_step": "idle",
    "workflow_completed": False,
    "workflow_experiment_started": False,
    "workflow_experiment_completed": False,
    "workflow_experiment_outputs": None,
    "research_goal": "",
}

DEFAULT_AGENT_COUNTS = {
    "hypothesis": 0,
    "experiment": 0,
    "curve_fit": 0,
    "analysis": 0,
    "router": 0,
    "watcher": 0,
}


def _get_conn() -> sqlite3.Connection:
    """Get thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = get_db_path()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(db_path, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


class DatabaseManager:
    """Manages SQLite persistence for app state."""

    def __init__(self):
        self._conn = None

    def _conn(self) -> sqlite3.Connection:
        return _get_conn()

    def init_schema(self) -> None:
        """Create tables if they do not exist."""
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS app_config (
                key TEXT PRIMARY KEY,
                value_text TEXT,
                value_int INTEGER,
                value_real REAL,
                value_json TEXT
            );

            CREATE TABLE IF NOT EXISTS experimental_constraints (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                techniques_json TEXT,
                equipment_json TEXT,
                parameters_json TEXT,
                focus_areas_json TEXT,
                liquid_handling_json TEXT
            );

            CREATE TABLE IF NOT EXISTS jupyter_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                server_url TEXT,
                token TEXT,
                upload_enabled INTEGER,
                notebook_path TEXT
            );

            CREATE TABLE IF NOT EXISTS conversation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                mode TEXT NOT NULL,
                prompt_session_id TEXT,
                timestamp TEXT NOT NULL,
                payload_json TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_usage_counts (
                agent_name TEXT PRIMARY KEY,
                count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TEXT,
                ml_model_choice TEXT
            );

            CREATE TABLE IF NOT EXISTS workflow_steps (
                workflow_id INTEGER NOT NULL,
                step_order INTEGER NOT NULL,
                name TEXT NOT NULL,
                automatic INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (workflow_id, step_order),
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            );

            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state_json TEXT
            );
        """)
        conn.commit()

    def ensure_defaults(self) -> None:
        """Ensure all tables have default values where missing."""
        conn = _get_conn()
        self.init_schema()

        # Insert app_config defaults
        for key, value in DEFAULT_APP_CONFIG.items():
            if value is None:
                continue
            cur = conn.execute(
                "SELECT 1 FROM app_config WHERE key = ?", (key,)
            )
            if cur.fetchone() is None:
                self._set_config_value(conn, key, value)

        # experimental_constraints
        cur = conn.execute("SELECT 1 FROM experimental_constraints WHERE id = 1")
        if cur.fetchone() is None:
            conn.execute(
                """INSERT INTO experimental_constraints (id, techniques_json, equipment_json,
                   parameters_json, focus_areas_json, liquid_handling_json)
                   VALUES (1, ?, ?, ?, ?, ?)""",
                (
                    json.dumps(DEFAULT_EXPERIMENTAL_CONSTRAINTS["techniques"]),
                    json.dumps(DEFAULT_EXPERIMENTAL_CONSTRAINTS["equipment"]),
                    json.dumps(DEFAULT_EXPERIMENTAL_CONSTRAINTS["parameters"]),
                    json.dumps(DEFAULT_EXPERIMENTAL_CONSTRAINTS["focus_areas"]),
                    json.dumps(DEFAULT_EXPERIMENTAL_CONSTRAINTS["liquid_handling"]),
                ),
            )

        # jupyter_config
        cur = conn.execute("SELECT 1 FROM jupyter_config WHERE id = 1")
        if cur.fetchone() is None:
            jc = DEFAULT_JUPYTER_CONFIG
            conn.execute(
                """INSERT INTO jupyter_config (id, server_url, token, upload_enabled, notebook_path)
                   VALUES (1, ?, ?, ?, ?)""",
                (jc["server_url"], jc["token"], int(jc["upload_enabled"]), jc["notebook_path"]),
            )

        # agent_usage_counts
        for agent, count in DEFAULT_AGENT_COUNTS.items():
            cur = conn.execute(
                "SELECT 1 FROM agent_usage_counts WHERE agent_name = ?", (agent,)
            )
            if cur.fetchone() is None:
                conn.execute(
                    "INSERT INTO agent_usage_counts (agent_name, count) VALUES (?, ?)",
                    (agent, count),
                )

        # session_state (single row)
        cur = conn.execute("SELECT 1 FROM session_state WHERE id = 1")
        if cur.fetchone() is None:
            state = dict(DEFAULT_SESSION_STATE)
            state["current_prompt_session_id"] = str(uuid.uuid4())
            state["start_time"] = datetime.now().timestamp()
            conn.execute(
                "INSERT INTO session_state (id, state_json) VALUES (1, ?)",
                (json.dumps(state),),
            )

        conn.commit()

    def _set_config_value(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        """Store a config value in app_config."""
        if isinstance(value, bool):
            conn.execute(
                "INSERT OR REPLACE INTO app_config (key, value_int) VALUES (?, ?)",
                (key, 1 if value else 0),
            )
        elif isinstance(value, int):
            conn.execute(
                "INSERT OR REPLACE INTO app_config (key, value_int) VALUES (?, ?)",
                (key, value),
            )
        elif isinstance(value, float):
            conn.execute(
                "INSERT OR REPLACE INTO app_config (key, value_real) VALUES (?, ?)",
                (key, value),
            )
        elif isinstance(value, (dict, list)):
            conn.execute(
                "INSERT OR REPLACE INTO app_config (key, value_json) VALUES (?, ?)",
                (key, json.dumps(value)),
            )
        else:
            conn.execute(
                "INSERT OR REPLACE INTO app_config (key, value_text) VALUES (?, ?)",
                (key, str(value) if value is not None else ""),
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable by key. Routes to appropriate table."""
        conn = _get_conn()

        # app_config keys
        if key in DEFAULT_APP_CONFIG:
            row = conn.execute(
                "SELECT value_text, value_int, value_real, value_json FROM app_config WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return DEFAULT_APP_CONFIG.get(key, default)
            for i, col in enumerate(["value_text", "value_int", "value_real", "value_json"]):
                v = row[i]
                if v is not None:
                    if col == "value_json":
                        try:
                            return json.loads(v)
                        except json.JSONDecodeError:
                            return v
                    if col == "value_int" and key == "experimental_mode":
                        return bool(v)
                    return v
            return default

        # experimental_constraints
        if key == "experimental_constraints":
            row = conn.execute(
                "SELECT techniques_json, equipment_json, parameters_json, focus_areas_json, liquid_handling_json FROM experimental_constraints WHERE id = 1"
            ).fetchone()
            if row is None:
                return DEFAULT_EXPERIMENTAL_CONSTRAINTS
            return {
                "techniques": json.loads(row[0] or "[]"),
                "equipment": json.loads(row[1] or "[]"),
                "parameters": json.loads(row[2] or "[]"),
                "focus_areas": json.loads(row[3] or "[]"),
                "liquid_handling": json.loads(row[4] or "{}"),
            }

        # jupyter_config
        if key == "jupyter_config":
            row = conn.execute(
                "SELECT server_url, token, upload_enabled, notebook_path FROM jupyter_config WHERE id = 1"
            ).fetchone()
            if row is None:
                return DEFAULT_JUPYTER_CONFIG
            return {
                "server_url": row[0] or "",
                "token": row[1] or "",
                "upload_enabled": bool(row[2]) if row[2] is not None else False,
                "notebook_path": row[3] or "Automated Agent",
            }

        # workflows
        if key == "workflows":
            return self.get_workflows()

        # conversation_events (from table, not session_state)
        if key == "conversation_events":
            return self.get_conversation_events()

        # uploaded_files (from table for full list)
        if key == "uploaded_files":
            return self.get_uploaded_files()

        # agent_usage_counts
        if key == "agent_usage_counts":
            rows = conn.execute(
                "SELECT agent_name, count FROM agent_usage_counts"
            ).fetchall()
            return {r[0]: r[1] for r in rows} if rows else dict(DEFAULT_AGENT_COUNTS)

        # session_state (transient state) - exclude keys handled above
        if key in DEFAULT_SESSION_STATE or key in (
            "prompt_session", "current_workflow", "workflow_steps",
            "interactions", "conversation_history", "allow_followup",
            "selected_wells", "plate_format", "last_processed_trigger_time",
            "watcher_auto_triggered_file", "watcher_auto_trigger_time",
            "auto_triggered_results", "auto_triggered_results_file",
            "auto_run_curve_fitting", "auto_run_data_file", "auto_run_comp_file",
            "auto_run_params", "auto_ml_after_curve_fitting", "auto_route_to_analysis",
            "workflow_curve_fitting_files", "ml_auto_json_path", "ml_auto_csv_path",
            "ml_auto_composition_path", "manual_clarified_question",
            "manual_socratic_questions", "manual_socratic_answers",
            "manual_thoughts", "manual_hypothesis", "optimization_model_choice",
            "ml_model_config", "gp_results", "gp_results_available",
            "analysis_ready", "curve_fitting_results", "analysis_results",
            "analysis_recommendations", "gp_suggested_compositions",
            "analysis_full_report", "analysis_feedback", "next_agent",
            "pending_additional_question", "workflow_ml_model_choice",
            "workflow_curve_fitting_completed", "workflow_created_at",
            "demo_workflow_running", "editing",
        ):
            row = conn.execute(
                "SELECT state_json FROM session_state WHERE id = 1"
            ).fetchone()
            if row is None:
                return default
            state = json.loads(row[0] or "{}")
            if key in state:
                val = state[key]
                if key == "selected_wells" and isinstance(val, list):
                    return set(val)
                return val
            return default

        return default

    def set(self, key: str, value: Any) -> None:
        """Set a variable by key."""
        conn = _get_conn()

        if key in DEFAULT_APP_CONFIG:
            self._set_config_value(conn, key, value)
            conn.commit()
            return

        if key == "experimental_constraints":
            ec = value
            conn.execute(
                """INSERT OR REPLACE INTO experimental_constraints
                   (id, techniques_json, equipment_json, parameters_json, focus_areas_json, liquid_handling_json)
                   VALUES (1, ?, ?, ?, ?, ?)""",
                (
                    json.dumps(ec.get("techniques", [])),
                    json.dumps(ec.get("equipment", [])),
                    json.dumps(ec.get("parameters", [])),
                    json.dumps(ec.get("focus_areas", [])),
                    json.dumps(ec.get("liquid_handling", {})),
                ),
            )
            conn.commit()
            return

        if key == "jupyter_config":
            jc = value
            conn.execute(
                """INSERT OR REPLACE INTO jupyter_config (id, server_url, token, upload_enabled, notebook_path)
                   VALUES (1, ?, ?, ?, ?)""",
                (
                    jc.get("server_url", ""),
                    jc.get("token", ""),
                    int(jc.get("upload_enabled", False)),
                    jc.get("notebook_path", "Automated Agent"),
                ),
            )
            conn.commit()
            return

        if key == "agent_usage_counts":
            for agent, count in value.items():
                conn.execute(
                    "INSERT OR REPLACE INTO agent_usage_counts (agent_name, count) VALUES (?, ?)",
                    (agent, count),
                )
            conn.commit()
            return

        if key == "workflows" and isinstance(value, dict):
            for wf_name, wf_data in value.items():
                if isinstance(wf_data, dict):
                    self.save_workflow(
                        wf_name,
                        wf_data.get("steps", []),
                        wf_data.get("ml_model_choice"),
                    )
            return

        # session_state
        row = conn.execute("SELECT state_json FROM session_state WHERE id = 1").fetchone()
        state = json.loads(row[0] or "{}") if row else {}
        if key == "selected_wells" and isinstance(value, set):
            value = list(value)
        state[key] = value
        conn.execute(
            "UPDATE session_state SET state_json = ? WHERE id = 1",
            (json.dumps(state),),
        )
        conn.commit()

    def get_conversation_events(self) -> List[Dict]:
        """Get all conversation events."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT type, mode, prompt_session_id, timestamp, payload_json FROM conversation_events ORDER BY id"
        ).fetchall()
        return [
            {
                "type": r[0],
                "mode": r[1],
                "prompt_session_id": r[2],
                "timestamp": r[3],
                "payload": json.loads(r[4] or "{}"),
            }
            for r in rows
        ]

    def append_conversation_event(
        self, event_type: str, mode: str, prompt_session_id: str, payload: dict
    ) -> None:
        """Append a conversation event."""
        conn = _get_conn()
        conn.execute(
            """INSERT INTO conversation_events (type, mode, prompt_session_id, timestamp, payload_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event_type,
                mode,
                prompt_session_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                json.dumps(payload),
            ),
        )
        conn.commit()

    def get_workflows(self) -> Dict[str, Dict]:
        """Get all saved workflows."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, name, created_at, ml_model_choice FROM workflows"
        ).fetchall()
        result = {}
        for r in rows:
            wf_id, name, created_at, ml_model_choice = r
            steps = self.get_workflow_steps(wf_id)
            result[name] = {
                "name": name,
                "steps": steps,
                "ml_model_choice": ml_model_choice,
                "created_at": created_at or "",
            }
        return result

    def save_workflow(
        self, name: str, steps: List[Dict], ml_model_choice: Optional[str] = None
    ) -> None:
        """Save a workflow."""
        conn = _get_conn()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute("SELECT id FROM workflows WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            wf_id = row[0]
            conn.execute("DELETE FROM workflow_steps WHERE workflow_id = ?", (wf_id,))
            conn.execute(
                "UPDATE workflows SET created_at = ?, ml_model_choice = ? WHERE id = ?",
                (created_at, ml_model_choice or "", wf_id),
            )
        else:
            conn.execute(
                "INSERT INTO workflows (name, created_at, ml_model_choice) VALUES (?, ?, ?)",
                (name, created_at, ml_model_choice or ""),
            )
            wf_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for i, step in enumerate(steps):
            conn.execute(
                "INSERT INTO workflow_steps (workflow_id, step_order, name, automatic) VALUES (?, ?, ?, ?)",
                (wf_id, i, step.get("name", ""), int(step.get("automatic", False))),
            )
        conn.commit()

    def delete_workflow(self, name: str) -> None:
        """Delete a workflow by name."""
        conn = _get_conn()
        cur = conn.execute("SELECT id FROM workflows WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            wf_id = row[0]
            conn.execute("DELETE FROM workflow_steps WHERE workflow_id = ?", (wf_id,))
            conn.execute("DELETE FROM workflows WHERE id = ?", (wf_id,))
            conn.commit()

    def get_workflow_steps(self, workflow_id: int) -> List[Dict]:
        """Get steps for a workflow."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT name, automatic FROM workflow_steps WHERE workflow_id = ? ORDER BY step_order",
            (workflow_id,),
        ).fetchall()
        return [{"name": r[0], "automatic": bool(r[1])} for r in rows]

    def get_uploaded_files(self) -> List[Dict]:
        """Get uploaded file metadata."""
        conn = _get_conn()
        rows = conn.execute(
            "SELECT name, path, timestamp FROM uploaded_files ORDER BY id"
        ).fetchall()
        return [{"name": r[0], "path": r[1], "timestamp": r[2]} for r in rows]

    def add_uploaded_file(self, filename: str, path: str) -> None:
        """Add uploaded file metadata."""
        conn = _get_conn()
        conn.execute(
            "INSERT INTO uploaded_files (name, path, timestamp) VALUES (?, ?, ?)",
            (filename, path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()

    def clear_session_state(self, keep_keys: Optional[List[str]] = None) -> None:
        """Clear session state, optionally keeping some keys."""
        keep_keys = keep_keys or []
        conn = _get_conn()
        row = conn.execute("SELECT state_json FROM session_state WHERE id = 1").fetchone()
        state = json.loads(row[0] or "{}") if row else {}
        new_state = {k: v for k, v in state.items() if k in keep_keys}
        conn.execute(
            "UPDATE session_state SET state_json = ? WHERE id = 1",
            (json.dumps(new_state),),
        )
        conn.commit()

    def clear_all_except(self, keep_keys: List[str]) -> None:
        """Clear app_config and session_state except for keep_keys."""
        conn = _get_conn()
        if "start_time" not in keep_keys:
            conn.execute("DELETE FROM app_config WHERE key = 'start_time'")
        for k in ["api_key", "api_key_source"]:
            if k not in keep_keys:
                conn.execute("DELETE FROM app_config WHERE key = ?", (k,))
        # Reset session_state to defaults
        state = dict(DEFAULT_SESSION_STATE)
        state["current_prompt_session_id"] = str(uuid.uuid4())
        state["start_time"] = datetime.now().timestamp()
        for k in keep_keys:
            if k in DEFAULT_APP_CONFIG:
                pass  # kept in app_config
        conn.execute(
            "UPDATE session_state SET state_json = ? WHERE id = 1",
            (json.dumps(state),),
        )
        conn.commit()
