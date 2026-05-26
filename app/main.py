import os
import sys
from dotenv import load_dotenv
from app.workflow import graph_app
from app.utils.logger import logger

# Load environment variables
load_dotenv()

def run_chat(prompt: str):
    """Execute the RAG workflow for a given prompt and return the result."""
    try:
        logger.info(f"Received prompt: {prompt}")
        state = {"prompt": prompt}
        # Support both sync and async graph execution
        if hasattr(graph_app, "invoke"):
            result = graph_app.invoke(state)
        else:
            # Assume async method
            import asyncio
            result = asyncio.run(graph_app.ainvoke(state))
        logger.info("Workflow execution completed")
        return {"status": "success", "response": result}
    except Exception as e:
        logger.error(f"Chat execution error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Simple CLI: prompt can be passed as an argument or entered interactively
    if len(sys.argv) > 1:
        user_prompt = sys.argv[1]
    else:
        user_prompt = input("Enter prompt: ")
    output = run_chat(user_prompt)
    print(output)