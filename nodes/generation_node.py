from groq import Groq
from config.settings import settings

# Initialize Groq client
llm_client = Groq(api_key=settings.groq_api_key)

def generation_node(state):
    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = f"""You are a precise and helpful AI assistant for college placement information.
Answer the user's question using ONLY the provided context. Do not use outside knowledge.

Context:
{context}

Question:
{state['query']}

Provide a concise, direct answer:"""

    response = llm_client.chat.completions.create(
        model=settings.generation.model,
        messages=[
            {"role": "system", "content": "You are a precise assistant that answers questions based only on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer
    }