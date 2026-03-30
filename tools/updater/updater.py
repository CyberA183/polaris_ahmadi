"""
GitHub Releases updater support for the packaged macOS app.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

from tools.paths import (
    get_current_app_bundle_path,
    get_resource_path,
    get_update_download_path,
    get_update_helper_runtime_path,
    get_update_pending_path,
    get_update_staging_dir,
    get_updater_log_path,
)

from .version import APP_VERSION, DEFAULT_MACOS_ASSET_NAME, VERSION_MANIFEST_URL


@dataclass
class UpdateInfo:
    version: str
    notes: str
    download_url: str


def get_current_version() -> str:
    """Return the current app version."""
    return APP_VERSION


def _version_key(version: str) -> tuple:
    parts = []
    for item in version.strip().lstrip("v").split("."):
        if item.isdigit():
            parts.append(int(item))
        else:
            parts.append(item)
    return tuple(parts)


def is_update_available(current_version: str, latest_version: str) -> bool:
    """Return True when the latest version is newer than the current version."""
    return _version_key(latest_version) > _version_key(current_version)


def fetch_latest_version_info(url: str = VERSION_MANIFEST_URL, timeout: int = 8) -> UpdateInfo:
    """Fetch version.json and return the latest macOS update info."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Polaris-Updater"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    version = payload["version"]
    notes = payload.get("notes", "")
    macos = payload.get("macos", {})
    download_url = macos.get("url")
    if not download_url:
        download_url = (
            f"https://github.com/CyberA183/polaris_ahmadi/releases/download/"
            f"v{version}/{DEFAULT_MACOS_ASSET_NAME}"
        )

    return UpdateInfo(version=version, notes=notes, download_url=download_url)


def write_updater_log(message: str) -> None:
    """Append a message to the updater log."""
    log_path = Path(get_updater_log_path())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message.rstrip() + "\n")


def clear_pending_update() -> None:
    """Remove any pending update marker."""
    pending_path = Path(get_update_pending_path())
    if pending_path.exists():
        pending_path.unlink()


def load_pending_update() -> dict | None:
    """Load pending update metadata if it exists."""
    pending_path = Path(get_update_pending_path())
    if not pending_path.exists():
        return None
    with open(pending_path, "r", encoding="utf-8") as pending_file:
        return json.load(pending_file)


def ensure_update_helper_script() -> str:
    """Copy the bundled update helper to a writable runtime location."""
    source = Path(get_resource_path("tools/updater/update_helper.py"))
    destination = Path(get_update_helper_runtime_path())
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return str(destination)


def stage_update(update: UpdateInfo, install_target: str | None = None) -> dict:
    """Download and extract an app update, then write a pending marker."""
    install_target = install_target or get_current_app_bundle_path()
    if not install_target:
        raise RuntimeError("Could not determine the installed app bundle path.")

    download_path = Path(get_update_download_path())
    staging_dir = Path(get_update_staging_dir())
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        update.download_url,
        headers={"User-Agent": "Polaris-Updater"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        download_path.write_bytes(response.read())

    with zipfile.ZipFile(download_path, "r") as archive:
        archive.extractall(staging_dir)

    staged_apps = sorted(staging_dir.rglob("*.app"))
    if not staged_apps:
        raise RuntimeError("Downloaded update archive did not contain a .app bundle.")

    staged_bundle = staged_apps[0]
    pending_data = {
        "current_version": get_current_version(),
        "target_version": update.version,
        "download_url": update.download_url,
        "notes": update.notes,
        "install_target": install_target,
        "staged_bundle_path": str(staged_bundle),
        "download_path": str(download_path),
        "staging_dir": str(staging_dir),
    }

    with open(get_update_pending_path(), "w", encoding="utf-8") as pending_file:
        json.dump(pending_data, pending_file, indent=2)

    return pending_data


def serialize_update_info(update: UpdateInfo | None) -> dict | None:
    """Convert update info to a JSON-serializable dict."""
    if update is None:
        return None
    return asdict(update)
