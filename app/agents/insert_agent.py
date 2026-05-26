from app.agents.state import AgentState

from app.pipeline.insert.prompt_builder import (
    PromptBuilder
)


def insert_node(
    state: AgentState
) -> AgentState:

    docs = state.get(
        "refined_docs",
        []
    )

    prompt = PromptBuilder().build(
        state["prompt"],
        docs
    )

    state["final_prompt"] = prompt

    return state