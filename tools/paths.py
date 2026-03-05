"""
Platform-specific paths for Polaris Ahmadi.
Supports both development and PyInstaller frozen executable.
"""

import os
import sys
from pathlib import Path

# Application name for user data directory
APP_NAME = "PolarisAhmadi"
DB_FILENAME = "polaris.db"


def is_frozen() -> bool:
    """Return True if running as a PyInstaller frozen executable."""
    return getattr(sys, "frozen", False)


def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to a resource. Works for both dev and PyInstaller.
    For frozen: resources are in sys._MEIPASS.
    For dev: relative to project root.
    """
    if is_frozen():
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = Path(__file__).parent.parent
    return str(Path(base) / relative_path)


def get_user_data_dir() -> str:
    """
    Get platform-specific user data directory for persistent storage.
    - Windows: %APPDATA%\\PolarisAhmadi
    - macOS: ~/Library/Application Support/PolarisAhmadi
    - Linux: ~/.local/share/polaris_ahmadi
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        path = Path(base) / APP_NAME
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        path = Path.home() / ".local" / "share" / "polaris_ahmadi"

    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_db_path() -> str:
    """Get path to the SQLite database file."""
    return str(Path(get_user_data_dir()) / DB_FILENAME)


def get_env_path() -> str:
    """Get path to .env file in user data dir (for packaged app)."""
    return str(Path(get_user_data_dir()) / ".env")
