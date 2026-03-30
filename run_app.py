#!/usr/bin/env python3
"""
Launcher for Polaris Ahmadi.
Supports both development (python run_app.py) and PyInstaller frozen executable.
Launches Streamlit in the background and displays it in a PyWebView native window.
Shows a loading screen immediately while the Streamlit server starts.
"""

import atexit
import html
import json
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

from tools.paths import (
    get_current_install_target_path,
    get_resource_path,
    get_update_pending_path,
    get_updater_log_path,
    get_user_data_dir,
)
from tools.updater import (
    fetch_latest_version_info,
    get_current_version,
    is_update_available,
    load_pending_update,
    stage_update,
    write_updater_log,
)
from tools.updater.version import APP_VERSION, VERSION_MANIFEST_URL

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

    if module_name == "tools.updater.update_helper":
        sys.argv = ["tools.updater.update_helper"] + module_args
        runpy.run_module("tools.updater.update_helper", run_name="__main__")
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


def build_message_html(title: str, body_html: str) -> str:
    safe_title = html.escape(title)
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
        h1 {{ font-size: 1.2rem; margin-bottom: 0.75rem; }}
        p, li {{ line-height: 1.45; margin-bottom: 0.65rem; }}
        code {{
            background: #e2e8f0;
            padding: 0.15rem 0.35rem;
            border-radius: 6px;
        }}
        ul {{ padding-left: 1.25rem; }}
    </style>
</head>
<body>
    <h1>{safe_title}</h1>
    {body_html}
</body>
</html>"""


def build_about_html(log_path: Path, url: str, update_summary: str) -> str:
    safe_log_path = html.escape(str(log_path))
    safe_url = html.escape(url)
    safe_update_summary = html.escape(update_summary)
    return build_message_html(
        f"About {APP_TITLE}",
        (
            f"<p>Version: <code>{APP_VERSION}</code></p>"
            f"<p>Native desktop launcher for the Polaris Streamlit application.</p>"
            f"<p>Startup log: <code>{safe_log_path}</code></p>"
            f"<p>Server URL: <code>{safe_url}</code></p>"
            f"<p>Updater status: {safe_update_summary}</p>"
        ),
    )


def build_error_html(message: str, log_path: Path) -> str:
    safe_message = html.escape(message)
    safe_log_path = html.escape(str(log_path))
    return build_message_html(
        "Failed to start Polaris",
        (
            f"<p>{safe_message}</p>"
            f"<p>Check the launcher log for details: <code>{safe_log_path}</code></p>"
        ),
    )


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


def is_updater_supported() -> bool:
    """Return True when the app can self-update on this platform."""
    return sys.platform in {"darwin", "win32"} and getattr(sys, "frozen", False) and bool(
        get_current_install_target_path()
    )


def launch_pending_update_helper() -> bool:
    """Launch the standalone updater helper for a pending update."""
    pending_update = load_pending_update()
    if not pending_update:
        return False

    write_updater_log("Launching updater helper")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.updater.update_helper",
            get_update_pending_path(),
            str(os.getpid()),
            get_updater_log_path(),
        ],
        start_new_session=True,
    )
    return True


def maybe_apply_pending_update() -> bool:
    """Hand off to the updater helper before normal app startup."""
    if not is_updater_supported():
        return False
    if not load_pending_update():
        return False

    log_message("Pending update detected; handing off to updater helper.")
    if launch_pending_update_helper():
        return True
    return False


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
    if maybe_apply_pending_update():
        return

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
    update_lock = threading.Lock()
    updater_state = {
        "checking": False,
        "status": "Updates unavailable",
        "latest_info": None,
        "pending_update": load_pending_update(),
        "last_error": None,
    }

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

    window = webview.create_window(APP_TITLE, html=LOADING_HTML)

    def show_info_window(title: str, message: str) -> None:
        webview.create_window(
            title,
            html=build_message_html(title, f"<p>{html.escape(message)}</p>"),
            width=440,
            height=240,
            resizable=False,
        )

    def update_summary() -> str:
        pending = updater_state.get("pending_update")
        if pending:
            return f"Downloaded {pending.get('target_version', 'update')} and ready to apply."
        latest = updater_state.get("latest_info")
        if latest:
            return f"Update available: {latest.version}"
        return updater_state.get("status", "No update information")

    def perform_update_check(show_result: bool = False) -> None:
        if not is_updater_supported():
            updater_state["status"] = "Updater available only in packaged desktop builds."
            return

        with update_lock:
            if updater_state["checking"]:
                return
            updater_state["checking"] = True

        try:
            latest_info = fetch_latest_version_info(VERSION_MANIFEST_URL)
            if is_update_available(get_current_version(), latest_info.version):
                updater_state["latest_info"] = latest_info
                updater_state["status"] = f"Update {latest_info.version} is available."
                log_message(updater_state["status"])
                if show_result:
                    show_info_window(
                        "Update Available",
                        (
                            f"Polaris {latest_info.version} is available.\n"
                            "Use Help > Download Latest Update to stage it for the next launch."
                        ),
                    )
            else:
                updater_state["latest_info"] = None
                updater_state["status"] = "You are running the latest version."
                if show_result:
                    show_info_window("No Updates", updater_state["status"])
        except Exception as exc:
            updater_state["latest_info"] = None
            updater_state["last_error"] = str(exc)
            updater_state["status"] = "Update check failed."
            log_message(f"Update check failed: {exc}")
            if show_result:
                show_info_window("Update Check Failed", str(exc))
        finally:
            updater_state["checking"] = False

    def download_latest_update() -> None:
        if not is_updater_supported():
            show_info_window(
                "Updater Unavailable",
                "Update download is only available in packaged desktop builds.",
            )
            return

        latest_info = updater_state.get("latest_info")
        if latest_info is None:
            perform_update_check(show_result=False)
            latest_info = updater_state.get("latest_info")

        if latest_info is None:
            show_info_window("No Update Available", updater_state.get("status", "No update found."))
            return

        try:
            pending_update = stage_update(
                latest_info,
                install_target=get_current_install_target_path(),
            )
            updater_state["pending_update"] = pending_update
            updater_state["status"] = (
                f"Downloaded {latest_info.version}. It will be installed on the next launch."
            )
            write_updater_log(
                f"Staged update {latest_info.version} from {latest_info.download_url}"
            )
            show_info_window(
                "Update Downloaded",
                (
                    f"Polaris {latest_info.version} has been downloaded.\n"
                    "Close and reopen the app to install it, or use Help > Apply Downloaded Update Now."
                ),
            )
        except Exception as exc:
            updater_state["last_error"] = str(exc)
            updater_state["status"] = "Update download failed."
            log_message(f"Update download failed: {exc}")
            show_info_window("Update Download Failed", str(exc))

    def apply_downloaded_update_now() -> None:
        if not updater_state.get("pending_update"):
            show_info_window("No Downloaded Update", "There is no staged update to apply.")
            return

        if launch_pending_update_helper():
            exit_app()
        else:
            show_info_window(
                "Updater Failed",
                "Could not launch the updater helper. Check the updater log for details.",
            )

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
            html=build_about_html(log_path, url, update_summary()),
            width=460,
            height=300,
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

    if is_updater_supported():
        updater_state["status"] = "Checking for updates..."
        threading.Thread(
            target=perform_update_check,
            kwargs={"show_result": False},
            daemon=True,
        ).start()
    else:
        updater_state["status"] = "Updater available only in packaged desktop builds."

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
                wm.MenuAction("Check for Updates", lambda: threading.Thread(
                    target=perform_update_check,
                    kwargs={"show_result": True},
                    daemon=True,
                ).start()),
                wm.MenuAction("Download Latest Update", lambda: threading.Thread(
                    target=download_latest_update,
                    daemon=True,
                ).start()),
                wm.MenuAction("Apply Downloaded Update Now", apply_downloaded_update_now),
                wm.MenuSeparator(),
                wm.MenuAction(f"About {APP_TITLE}", show_about),
            ],
        ),
    ]

    webview.start(launch_streamlit, window, menu=menu_items)
    kill_server(proc_holder["proc"])


if __name__ == "__main__":
    main()
