from typing import Dict, Any, Optional

import streamlit as st
from agents.base import BaseAgent
from tools.memory import MemoryManager


class FallbackAgent(BaseAgent):
    """
    Fallback agent that handles routing failures and provides error recovery.
    This agent is called when:
    - No other agent has sufficient confidence to handle a request
    - An agent fails during execution
    - Manual workflow is exhausted
    """
    
    def __init__(self, name: str = "Fallback Agent", desc: Optional[str] = None):
        super().__init__(name, desc or "Handles routing failures and provides error recovery")
        self.memory = MemoryManager()

    def confidence(self, payload: Dict[str, Any]) -> float:
        """
        Fallback agent has low confidence by default - it should only be used
        when no other agent can handle the request.
        """
        # Only confident if explicitly requested as fallback
        if payload.get("force_fallback") or payload.get("error_occurred"):
            return 1.0
        return 0.0

    def run_agent(self, memory: MemoryManager) -> None:
        """
        Handle fallback scenarios - provide user feedback and recovery options.
        """
        # Log the fallback event
        memory.log_event(
            event_type="fallback",
            payload={
                "message": "Fallback agent activated",
                "reason": "No suitable agent found or routing error occurred"
            },
            mode="fallback"
        )
        
        # Display error message and recovery options
        st.error("⚠️ **Routing Error: Unable to determine next agent**")
        
        st.markdown("""
        The system was unable to route your request to an appropriate agent. 
        This can happen when:
        - No agent has sufficient confidence to handle the current state
        - The manual workflow has been exhausted
        - An error occurred during agent execution
        """)
        
        # Provide recovery options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retry Routing", use_container_width=True):
                st.session_state.workflow_index = 0
                st.rerun()
        
        with col2:
            if st.button("🏠 Go to Home", use_container_width=True):
                st.switch_page("pages/home.py")
        
        with col3:
            if st.button("⚙️ Check Settings", use_container_width=True):
                st.switch_page("pages/settings.py")
        
        # Show current state for debugging
        with st.expander("🔍 Debug Information", expanded=False):
            st.json({
                "routing_mode": st.session_state.get("routing_mode", "Unknown"),
                "workflow_index": st.session_state.get("workflow_index", 0),
                "manual_workflow": st.session_state.get("manual_workflow", []),
                "last_hypothesis": st.session_state.get("last_hypothesis") is not None,
                "experimental_outputs": st.session_state.get("experimental_outputs") is not None,
                "uploaded_files": len(st.session_state.get("uploaded_files", []))
            })
        
        st.info("💡 **Tip:** Try adjusting your routing mode in Settings or ensure you have completed the prerequisite steps for your workflow.")