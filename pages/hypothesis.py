import streamlit as st
from tools.memory import MemoryManager
from agents.hypothesis_agent import HypothesisAgent

memory = MemoryManager()
memory.init_session()

def clear_conversation():
    memory.set_var("stage", "initial")
    memory.set_var("conversation_history", [])
    memory.set_var("allow_followup", False)
    st.toast("Conversation restarted")

def stop_and_create_hypothesis():
    memory.set_var("stop_hypothesis", True)
    memory.set_var("stage", "hypothesis")
    st.toast("Generating hypothesis...")
    st.rerun()

def go_back_stage():
    interactions = memory.get_var("interactions", [])
    if interactions:
        interactions.pop()
        memory.set_var("interactions", interactions)
        st.toast("Returned to previous stage")
    else:
        st.warning("No previous stage to go back to.")

st.set_page_config(layout="centered")

col_title, col_new = st.columns([5, 1])
with col_title:
    st.title("🧠 AI Hypothesis Agent")
with col_new:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ New", help="Clear conversation and start fresh", use_container_width=True):
        clear_conversation()
        st.rerun()

# Add custom CSS for better visual distinction between chat sections
st.markdown("""
    <style>
    /* User messages - light blue background */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageUser"]) {
        background-color: #E3F2FD !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        border-left: 4px solid #2196F3 !important;
    }

    /* Assistant messages - light green background */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAssistant"]) {
        background-color: #F1F8E9 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        margin: 1rem 0 !important;
        border-left: 4px solid #4CAF50 !important;
    }

    /* Add extra spacing between chat messages */
    div[data-testid="stChatMessage"] {
        margin-bottom: 1.5rem !important;
    }

    /* Style for section headers (bold text) */
    div[data-testid="stChatMessage"] strong {
        color: #1565C0 !important;
        font-size: 1.05em;
    }

    /* Visual separator for horizontal rules */
    div[data-testid="stChatMessage"] hr {
        border: none;
        border-top: 2px solid #BDBDBD;
        margin: 1rem 0;
    }

    /* Style for option numbers to make them stand out */
    div[data-testid="stChatMessage"] p strong:first-child {
        color: #7B1FA2 !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Layout styling
st.markdown("""
<style>
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {
        align-items: flex-end;
    }
    .bottom-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        border-top: 1px solid #ddd;
        padding: 0.8rem 1.5rem;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.05);
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# Display existing chat
chat_container = st.container()
with chat_container:
    interactions = memory.get_var("interactions", [])
    for i in interactions:
        with st.chat_message(i["role"]):
            # Add section headers for specific components to make them clear
            if i.get("component") == "socratic_answers":
                st.markdown("**Socratic Reasoning (LLM Answers to Its Own Questions):**")
            st.markdown(i["message"])

st.markdown("<br>", unsafe_allow_html=True)

# Bottom controls
bottom = st.container()
with bottom:
    with st.popover("Options"):
        st.markdown("#### Conversation Controls")
        st.button("Restart", use_container_width=True, on_click=clear_conversation)
        st.button("Stop & Create Hypothesis", use_container_width=True, on_click=stop_and_create_hypothesis)
        st.button("Go Back", use_container_width=True, on_click=go_back_stage)

# Initialize and run the Hypothesis Agent
# Get initial question from session state or use a default
initial_question = memory.get_var("initial_question", "")

# Create and run the agent
agent = HypothesisAgent(
    name="Hypothesis Agent",
    desc="Helps generate scientific hypotheses through Socratic questioning",
    question=initial_question
)

# Run the agent - it will handle all the UI and state management
agent.run_agent(memory)

# Workflow transition: offer manual Continue (no auto-switch)
if (
    memory.get_var("workflow_active")
    and memory.get_var("stage") == "analysis"
    and memory.get_var("hypothesis_ready")
):
    memory.set_var("workflow_step", "experiment")
    memory.set_var("workflow_experiment_started", False)
    st.divider()
    if st.button("Continue to Experiment Agent →", type="primary", use_container_width=True, key="hyp_continue_experiment"):
        st.switch_page("pages/experiment.py")
