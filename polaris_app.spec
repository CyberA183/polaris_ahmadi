# -*- mode: python ; coding: utf-8 -*-
# Deprecated: retained temporarily during Briefcase migration.

import os
import sys

block_cipher = None

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    ['streamlit_app_clean.py'],
    pathex=[current_dir],
    binaries=[],
    datas=[
        # Include all necessary files
        ('pages', 'pages'),
        ('agents', 'agents'),
        ('tools', 'tools'),
        ('data', 'data'),
        ('requirements.txt', '.'),
        ('README.md', '.'),
        ('WORKFLOW_TRANSCRIPT.md', '.'),
        # Include streamlit config
        (os.path.join(os.path.expanduser('~'), '.streamlit', 'config.toml'), '.streamlit'),
    ],
    hiddenimports=[
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'plotly',
        'lmfit',
        'PIL',
        'scipy',
        'sklearn',
        'openpyxl',
        'kaleido',
        # Add any other imports your app uses
        'streamlit.web.cli',
        'streamlit.runtime',
        'streamlit.components',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='POLARIS_Hypothesis_Agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',  # You'll need to create/add an icon file
)