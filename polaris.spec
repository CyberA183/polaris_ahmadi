# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None
project_root = Path(SPECPATH).resolve()
version_namespace = {}
exec(
    (project_root / "tools" / "updater" / "version.py").read_text(encoding="utf-8"),
    version_namespace,
)
APP_VERSION = version_namespace["APP_VERSION"]

datas = [
    (str(project_root / "streamlit_app.py"), "."),
    (str(project_root / "pages"), "pages"),
    (str(project_root / "agents"), "agents"),
    (str(project_root / "tools"), "tools"),
    (str(project_root / "watcher"), "watcher"),
    (str(project_root / "data"), "data"),
    (str(project_root / "app_icon"), "app_icon"),
]
datas += collect_data_files("streamlit")
datas += collect_data_files("sklearn")
datas += copy_metadata("streamlit")
datas += copy_metadata("pywebview")

hiddenimports = sorted(
    set(
        [
            "streamlit",
            "streamlit.web.cli",
            "streamlit.runtime.scriptrunner.magic_funcs",
            "webview",
            "webview.menu",
            "watcher.server",
            "watchdog",
            "watchdog.events",
            "watchdog.observers",
            "lmfit",
            "sklearn",
            "sklearn.gaussian_process",
            "pandas",
            "numpy",
            "scipy",
            "scipy.signal",
            "scipy.ndimage",
            "scipy.stats",
            "matplotlib",
            "plotly",
            "plotly.graph_objects",
            "altair",
            "openpyxl",
            "sqlite3",
            "google.generativeai",
            "dotenv",
            "PIL",
            "openai",
            "edison_client",
            "edison_client.models",
            "uvicorn",
            "uvicorn.logging",
            "uvicorn.loops.auto",
            "uvicorn.protocols.http.auto",
            "uvicorn.protocols.websockets.auto",
            "fastapi",
            "reportlab",
            "reportlab.lib",
            "reportlab.pdfgen",
            "tools.updater",
            "tools.updater.update_helper",
            "tools.updater.updater",
            "tools.updater.version",
        ]
        + collect_submodules("sklearn")
        + collect_submodules("webview", filter=lambda name: "android" not in name)
        + collect_submodules("watchdog")
    )
)

excludes = [
    "tkinter",
    "futurehouse_client",
    "IPython",
    "jedi",
    "pytest",
    "matplotlib.tests",
    "numpy.tests",
    "scipy.tests",
    "pandas.tests",
]

a = Analysis(
    ["run_app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Polaris",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Polaris",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Polaris.app",
        bundle_identifier="com.polarisahmadi.polaris",
        info_plist={
            "CFBundleName": "Polaris",
            "CFBundleDisplayName": "Polaris",
            "CFBundleVersion": APP_VERSION,
            "CFBundleShortVersionString": APP_VERSION,
            "NSHighResolutionCapable": True,
        },
    )