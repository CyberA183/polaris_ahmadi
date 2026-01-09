from typing import Any, Dict, Optional, Callable
import os

from agents.base import BaseAgent
from tools.memory import MemoryManager

# Lazy import for HTTP client and socratic
_requests = None
_socratic = None

def _lazy_import_requests():
    """Lazy import requests module"""
    global _requests
    if _requests is None:
        import requests
        _requests = requests
    return _requests

def _lazy_import_socratic():
    """Lazy import socratic module"""
    global _socratic
    if _socratic is None:
        from tools import socratic
        _socratic = socratic
    return _socratic

def _lazy_import_instruct():
    """Lazy import instruct module"""
    from tools.instruct import WATCHER_ROUTING_INSTRUCTIONS
    return WATCHER_ROUTING_INSTRUCTIONS


class WatcherAgent(BaseAgent):
    """
    Supervisor agent that reacts to filesystem signals and decides which
    downstream agent should run next via the router.

    This agent communicates with the FastAPI watcher server to handle
    filesystem events and uses LLM-based confidence routing.
    """

    def __init__(self, name: str = "Watcher Agent", desc: Optional[str] = None,
                 router_factory: Optional[Callable[[], Any]] = None,
                 server_url: Optional[str] = None):
        super().__init__(name, desc or "Supervises file events and triggers next agent.")
        self.router_factory = router_factory
        self.server_url = server_url or os.environ.get("WATCHER_SERVER_URL", "http://localhost:8000")

    def confidence(self, params: Dict[str, Any]) -> float:
        """
        Use LLM-based confidence scoring for filesystem events.
        The LLM analyzes the event context and determines confidence.
        """
        if params.get("source") != "filesystem":
            return 0.0
        
        try:
            # Lazy import socratic
            socratic = _lazy_import_socratic()
            instructions = _lazy_import_instruct()
            
            # Get context from params
            trigger_file = params.get("trigger_file", "")
            event_type = params.get("event", "created")
            metadata = params.get("metadata", {})
            
            # Build context for LLM
            # Note: In a non-Streamlit context, we can't access session_state
            # So we use what's available in the payload
            uploaded_files = params.get("uploaded_files", [])
            last_hypothesis = params.get("last_hypothesis")
            experimental_outputs = params.get("experimental_outputs")
            experimental_constraints = params.get("experimental_constraints", {})
            
            # Get available agents (would need to be passed in or retrieved)
            agent_names = params.get("available_agents", [
                "Hypothesis Agent",
                "Experiment Agent", 
                "Curve Fitting",
                "Watcher Agent"
            ])
            
            # Build prompt
            prompt = instructions.format(
                agent_names="\n".join([f"- {name}" for name in agent_names]),
                event_description=f"File {event_type}: {trigger_file}",
                trigger_file=trigger_file,
                uploaded_files=str(uploaded_files) if uploaded_files else "None",
                last_hypothesis=str(last_hypothesis)[:200] if last_hypothesis else "None",
                experimental_outputs=str(experimental_outputs)[:200] if experimental_outputs else "None",
                experimental_constraints=str(experimental_constraints)[:200] if experimental_constraints else "None"
            )
            
            # Ask LLM for confidence (simplified - returns 1.0 if LLM can make decision)
            try:
                response = socratic.generate_text_with_llm(prompt)
                # If LLM responds with an agent name, we're confident
                if response and any(name.lower() in response.lower() for name in agent_names):
                    return 1.0
            except Exception as e:
                # If LLM fails, fall back to simple heuristic
                pass
            
            # Fallback: simple heuristic based on file pattern
            if trigger_file:
                filename = os.path.basename(trigger_file)
                if filename.startswith("output_from_") and filename.endswith(".json"):
                    return 0.8  # High confidence for known pattern
                return 0.5  # Medium confidence for other files
            
            return 0.3  # Low confidence if no clear signal
            
        except Exception as e:
            # Fallback to simple heuristic
            return 1.0 if params.get("source") == "filesystem" else 0.0

    def run_agent(self, memory: MemoryManager) -> None:
        """
        Handle a single filesystem event by communicating with the FastAPI server.
        
        The server handles the actual filesystem watching, and this agent
        processes events and routes them to the appropriate next agent.
        """
        # Log that watcher agent handled the event
        memory.log_event(
            event_type="watcher",
            payload={"note": "filesystem event handled by WatcherAgent"},
            mode="watcher",
        )
        
        # In a Streamlit context, we might want to show UI feedback
        # In a background context, we just log
        try:
            import streamlit as st
            st.info("Watcher Agent detected a file change. Routing to next agent...")
        except (ImportError, RuntimeError):
            # Not in Streamlit context - just log
            pass