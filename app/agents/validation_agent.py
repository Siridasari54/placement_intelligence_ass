'''"""Validation Agent for LangGraph workflow.

Placeholder node that would check the generated answer for consistency,
groundedness, or any business rules. Here it simply sets a ``validation`` flag
to ``True``.
"""'''

from typing import Any
from app.agents.state import AgentState


def validation_node(state: AgentState) -> AgentState:
    """Simple validation placeholder.

    In a real system this could invoke a LLM or rule‑based checker. We set the
    ``validation`` field to ``True`` to indicate a pass.
    """
    state["validation"] = True
    return state
