"""
PyInstaller runtime hook — runs before any frozen module is imported.
Sets PYTHONNET_RUNTIME=coreclr so that clr_loader uses the .NET Core host API
instead of the legacy .NET Framework COM loader (ICorRuntimeHost / netfx),
which cannot resolve Python.Runtime.dll in a frozen bundle.
"""
import os

os.environ.setdefault("PYTHONNET_RUNTIME", "coreclr")
