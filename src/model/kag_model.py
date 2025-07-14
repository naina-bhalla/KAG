class KagModel:
    def __init__(self, llm):
        self.llm = llm

    def generate_answer(self, context: list) -> str:
        triples_str = "\n".join([f"- {t}" for t in context])
        prompt = f"""
You are a helpful assistant that answers medical questions using the provided knowledge triples.
Each triple is of the form (subject, predicate, object).

Knowledge Triples:
{triples_str}

Based on these triples, provide a clear answer in natural language.
Be concise and medically accurate.

Answer:
"""
        return self.llm(prompt).strip()
