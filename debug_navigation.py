#!/usr/bin/env python3
"""
Debug script to test navigation functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    import streamlit as st
    print(f"Streamlit version: {st.__version__}")

    # Try to create a simple page
    try:
        home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
        print("[OK] Home page creation successful")
    except Exception as e:
        print(f"[ERROR] Home page creation failed: {e}")
        import traceback
        traceback.print_exc()

    # Try navigation with just home
    try:
        pg = st.navigation({
            "General": [home],
        })
        print("[OK] Navigation creation successful")

        # Try to run it
        try:
            pg.run()
            print("[OK] Navigation run successful")
        except Exception as e:
            print(f"[ERROR] Navigation run failed: {e}")

    except Exception as e:
        print(f"[ERROR] Navigation creation failed: {e}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()