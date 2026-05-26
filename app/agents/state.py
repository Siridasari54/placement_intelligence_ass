from typing import TypedDict
from typing import List
from typing import Dict
from typing import Any


class AgentState(TypedDict, total=False):

    prompt: str

    rewritten_prompt: str

    route: str

    retrieved_docs: List[Any]

    reranked_docs: List[Any]

    refined_docs: List[Any]

    final_prompt: str

    response: str

    citations: List[str]

    validation: bool

    fallback: bool

    evaluation: Dict[str, Any]