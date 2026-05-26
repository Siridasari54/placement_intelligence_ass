from app.pipeline.generate.groq_llm import GroqLLM


class ResponseGenerator:

    def __init__(self):

        self.llm = GroqLLM()

    def generate(self, prompt):

        response = self.llm.generate(prompt)

        return response