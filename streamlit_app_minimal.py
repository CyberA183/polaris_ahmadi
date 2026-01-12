# Performance optimized imports
import streamlit as st
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.title("Minimal Test - Navigation Check")

# Try creating pages one by one
try:
    home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
    st.success("✓ Home page created successfully")
except Exception as e:
    st.error(f"✗ Home page failed: {e}")

try:
    dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon=":material/dashboard:")
    st.success("✓ Dashboard page created successfully")
except Exception as e:
    st.error(f"✗ Dashboard page failed: {e}")

try:
    settings = st.Page("pages/settings.py", title="Settings", icon=":material/settings:")
    st.success("✓ Settings page created successfully")
except Exception as e:
    st.error(f"✗ Settings page failed: {e}")

# Try navigation with working pages
working_pages = []
if 'home' in locals():
    working_pages.append(home)
if 'dashboard' in locals():
    working_pages.append(dashboard)
if 'settings' in locals():
    working_pages.append(settings)

if working_pages:
    try:
        pg = st.navigation({
            "Test": working_pages,
        })
        st.success("✓ Navigation created successfully")

        # Show current page info
        st.info(f"Navigation object created with {len(working_pages)} pages")

        # Run navigation
        pg.run()

    except Exception as e:
        st.error(f"✗ Navigation failed: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.error("No pages could be created successfully")

st.markdown("---")
st.markdown("If you see this message, the minimal navigation test is working!")