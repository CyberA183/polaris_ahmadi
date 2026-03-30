#!/usr/bin/env python3
"""
Launcher for Polaris Ahmadi.
Supports both development (python run_app.py) and PyInstaller frozen executable.
Launches Streamlit in the background and displays it in a PyWebView native window.
Shows a loading screen immediately while the Streamlit server starts.
"""

import atexit
import html
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import webview
import webview.menu as wm

from tools.paths import get_user_data_dir

APP_TITLE = "Polaris Ahmadi"
STREAMLIT_PORT = 8501
STARTUP_TIMEOUT_SECONDS = 30
SERVER_POLL_INTERVAL_SECONDS = 0.2


def _run_frozen_module_entrypoint() -> None:
    """Delegate -m execution back into the real module when frozen."""
    if not getattr(sys, "frozen", False) or len(sys.argv) < 3 or sys.argv[1] != "-m":
        return

    module_name = sys.argv[2]
    module_args = sys.argv[3:]

    if module_name == "streamlit":
        os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
        sys.argv = ["streamlit"] + module_args
        from streamlit.web import cli

        cli.main()
        sys.exit(0)

    if module_name == "watcher.server":
        sys.argv = ["watcher.server"] + module_args
        runpy.run_module("watcher.server", run_name="__main__")
        sys.exit(0)


_run_frozen_module_entrypoint()


LOADING_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #f5f7fb;
            color: #1f2937;
        }
        .card {
            min-width: 320px;
            padding: 2rem 2.25rem;
            background: white;
            border-radius: 16px;
            box-shadow: 0 12px 36px rgba(15, 23, 42, 0.10);
            text-align: center;
        }
        .spinner {
            width: 52px;
            height: 52px;
            margin: 0 auto 1.25rem auto;
            border: 4px solid #dbe4f0;
            border-top-color: #2563eb;
            border-radius: 50%;
            animation: spin 0.9s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        h1 {
            font-size: 1.45rem;
            font-weight: 650;
            margin-bottom: 0.45rem;
        }
        .subtitle {
            font-size: 0.95rem;
            color: #64748b;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="spinner"></div>
        <h1>Loading Polaris Ahmadi...</h1>
        <p class="subtitle">Starting Streamlit server</p>
    </div>
</body>
</html>"""


def build_about_html(log_path: Path, url: str) -> str:
    safe_log_path = html.escape(str(log_path))
    safe_url = html.escape(url)
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            background: #f8fafc;
            color: #0f172a;
            padding: 1.5rem;
        }}
        h1 {{ font-size: 1.25rem; margin-bottom: 0.5rem; }}
        p {{ margin-bottom: 0.6rem; line-height: 1.45; }}
        code {{
            background: #e2e8f0;
            padding: 0.15rem 0.35rem;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <h1>About Polaris Ahmadi</h1>
    <p>Native desktop launcher for the Polaris Streamlit application.</p>
    <p>Startup log: <code>{safe_log_path}</code></p>
    <p>Server URL: <code>{safe_url}</code></p>
</body>
</html>"""


def build_error_html(message: str, log_path: Path) -> str:
    safe_message = html.escape(message)
    safe_log_path = html.escape(str(log_path))
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f5f7fb;
            color: #1f2937;
            padding: 1.5rem;
        }}
        .card {{
            max-width: 520px;
            background: white;
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 12px 36px rgba(15, 23, 42, 0.10);
        }}
        h1 {{ font-size: 1.3rem; margin-bottom: 0.75rem; color: #b91c1c; }}
        p {{ line-height: 1.5; margin-bottom: 0.75rem; }}
        code {{
            background: #e2e8f0;
            padding: 0.15rem 0.35rem;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Failed to start Polaris</h1>
        <p>{safe_message}</p>
        <p>Check the launcher log for details: <code>{safe_log_path}</code></p>
    </div>
</body>
</html>"""


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to a resource. Works for both dev and PyInstaller frozen."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def get_log_path() -> Path:
    """Return the launcher log path in the user data directory."""
    try:
        return Path(get_user_data_dir()) / "launcher.log"
    except Exception:
        return Path.home() / ".polaris_launcher.log"


def log_message(message: str) -> None:
    """Append a timestamped message to the launcher log."""
    log_path = get_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"[{timestamp}] {message}\n")
    except OSError:
        pass


def kill_server(proc: subprocess.Popen | None) -> None:
    """Kill the Streamlit subprocess. Uses platform-specific logic."""
    if proc is None or proc.poll() is not None:
        return

    log_message(f"Stopping Streamlit subprocess PID={proc.pid}")
    try:
        if os.name == "nt":
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
        else:
            proc.kill()
    except Exception as exc:
        log_message(f"Failed to stop Streamlit cleanly: {exc}")


def wait_for_server(
    url: str,
    proc: subprocess.Popen | None = None,
    timeout_seconds: int = STARTUP_TIMEOUT_SECONDS,
    poll_interval: float = SERVER_POLL_INTERVAL_SECONDS,
) -> bool:
    """Poll until the server responds or timeout is reached."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            log_message(f"Streamlit exited early with code {proc.returncode}")
            return False

        try:
            with urllib.request.urlopen(url, timeout=0.5):
                return True
        except (OSError, urllib.error.URLError):
            time.sleep(poll_interval)

    return False


def main():
    app_path = get_resource_path("streamlit_app.py")
    if not os.path.exists(app_path):
        print(f"Error: streamlit_app.py not found at {app_path}")
        sys.exit(1)

    log_path = get_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    except OSError:
        pass

    port = STREAMLIT_PORT
    url = f"http://localhost:{port}"
    proc_holder = {"proc": None}
    startup_state = {"starting": False}
    startup_lock = threading.Lock()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]

    def launch_streamlit(target_window) -> None:
        with startup_lock:
            if startup_state["starting"]:
                return
            existing_proc = proc_holder["proc"]
            if existing_proc is not None and existing_proc.poll() is None:
                target_window.load_url(url)
                return
            startup_state["starting"] = True

        try:
            target_window.load_html(LOADING_HTML)
            env = os.environ.copy()
            env["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
            env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
            env["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

            log_message(f"Launching Streamlit from {app_path}")
            log_handle = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            log_handle.close()
            proc_holder["proc"] = proc
            atexit.register(kill_server, proc)

            if wait_for_server(url, proc=proc):
                log_message(f"Streamlit server ready at {url}")
                time.sleep(0.1)
                target_window.load_url(url)
            else:
                exit_code = proc.poll()
                if exit_code is None:
                    message = (
                        f"Streamlit did not start within {STARTUP_TIMEOUT_SECONDS} seconds."
                    )
                else:
                    message = f"Streamlit exited early with code {exit_code}."

                log_message(message)
                kill_server(proc)
                proc_holder["proc"] = None
                target_window.load_html(build_error_html(message, log_path))
        finally:
            with startup_lock:
                startup_state["starting"] = False

    window = webview.create_window(APP_TITLE, html=LOADING_HTML)

    def reload_app() -> None:
        active_window = webview.active_window() or window
        proc = proc_holder["proc"]
        if proc is not None and proc.poll() is None:
            active_window.load_url(url)
        else:
            threading.Thread(
                target=launch_streamlit,
                args=(active_window,),
                daemon=True,
            ).start()

    def show_loading_screen() -> None:
        active_window = webview.active_window() or window
        active_window.load_html(LOADING_HTML)

    def show_about() -> None:
        webview.create_window(
            f"About {APP_TITLE}",
            html=build_about_html(log_path, url),
            width=460,
            height=260,
            resizable=False,
        )

    def exit_app() -> None:
        kill_server(proc_holder["proc"])
        proc_holder["proc"] = None
        for existing_window in list(webview.windows):
            try:
                existing_window.destroy()
            except Exception:
                pass

    menu_items = [
        wm.Menu(
            "File",
            [
                wm.MenuAction("Reload App", reload_app),
                wm.MenuSeparator(),
                wm.MenuAction("Exit", exit_app),
            ],
        ),
        wm.Menu(
            "View",
            [
                wm.MenuAction("Reload Current Page", reload_app),
                wm.MenuAction("Show Loading Screen", show_loading_screen),
            ],
        ),
        wm.Menu(
            "Help",
            [
                wm.MenuAction(f"About {APP_TITLE}", show_about),
            ],
        ),
    ]

    webview.start(launch_streamlit, window, menu=menu_items)
    kill_server(proc_holder["proc"])


if __name__ == "__main__":
    main()
