from app.agents.state import AgentState

from app.pipeline.generate.response_generator import (
    ResponseGenerator
)


def summarizer_node(
    state: AgentState
) -> AgentState:

    prompt = state["final_prompt"]

    generator = ResponseGenerator()

    response = generator.generate(
        prompt
    )

    state["response"] = response

    return state