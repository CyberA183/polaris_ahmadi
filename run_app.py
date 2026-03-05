#!/usr/bin/env python3
"""
Launcher for Polaris Ahmadi.
Supports both development (python run_app.py) and PyInstaller frozen executable.
Launches Streamlit in the background and displays it in a PyWebView native window.
"""

import atexit
import os
import subprocess
import sys
import time
import urllib.request

import webview

# When frozen (PyInstaller), subprocess.Popen([sys.executable, "-m", "streamlit", ...])
# re-invokes this executable. The bootloader runs this script again with those args.
# We must detect that and delegate to Streamlit's CLI instead of running our launcher,
# otherwise we get an infinite loop of windows.
if getattr(sys, "frozen", False) and len(sys.argv) >= 3 and sys.argv[1] == "-m" and sys.argv[2] == "streamlit":
    sys.argv = ["streamlit"] + sys.argv[3:]
    from streamlit.web import cli
    cli.main()
    sys.exit(0)


def get_resource_path(relative_path: str) -> str:
    """Get absolute path to a resource. Works for both dev and PyInstaller frozen."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, relative_path)


def kill_server(proc: subprocess.Popen) -> None:
    """Kill the Streamlit subprocess. Uses platform-specific logic."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
        else:
            proc.kill()
    except Exception:
        pass


def wait_for_server(url: str, timeout_seconds: int = 30) -> bool:
    """Poll until the server responds or timeout is reached."""
    for _ in range(timeout_seconds):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except OSError:
            time.sleep(1)
    return False


def main():
    app_path = get_resource_path("streamlit_app.py")
    if not os.path.exists(app_path):
        print(f"Error: streamlit_app.py not found at {app_path}")
        sys.exit(1)

    port = 8501
    url = f"http://localhost:{port}"

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
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(kill_server, proc)

    if not wait_for_server(url):
        kill_server(proc)
        print("Error: Streamlit server failed to start within 30 seconds")
        sys.exit(1)

    window = webview.create_window("Polaris Ahmadi", url)
    webview.start()

    kill_server(proc)


if __name__ == "__main__":
    main()
