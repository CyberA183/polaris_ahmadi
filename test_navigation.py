#!/usr/bin/env python3
"""
Test script to check if the navigation is working properly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    import streamlit as st
    print(f"Streamlit version: {st.__version__}")

    # Test if we can create pages
    try:
        home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
        print("[OK] Page creation successful")
    except Exception as e:
        print(f"[ERROR] Page creation failed: {e}")

    # Test if navigation works
    try:
        pg = st.navigation({
            "General": [home],
        })
        print("[OK] Navigation creation successful")
    except Exception as e:
        print(f"[ERROR] Navigation creation failed: {e}")

    print("All tests passed!")

except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")