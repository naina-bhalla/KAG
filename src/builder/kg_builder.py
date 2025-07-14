from typing import List, Tuple
from src.llm.cohere import Cohere
import json
import re

class KnowledgeGraphBuilder:
    def __init__(self, llm: Cohere):
        self.llm = llm

    def _build_prompt(self, doc: str) -> str:
        return f"""
You are a medical assistant. Extract subject-predicate-object triples from the text below.
Normalize all medical terms (diseases, drugs, symptoms) to their canonical forms.
Only return the triples list in valid Python syntax. Do not include any explanation or text.

Example:
Text: "The patient was given Tylenol for a high temperature."
Triples:
- ("Paracetamol", "treats", "fever")
- ("patient", "given", "Paracetamol")

Text: "{doc}"
Triples:
"""

    def build_(self, docs: List[str]) -> Tuple[List[Tuple[str, str, str]], dict]:
        all_triples = []
        triple_map = {}

        for idx, doc in enumerate(docs):
            prompt = self._build_prompt(doc)
            response = self.llm(prompt)
            try:
                # Remove markdown/code fences
                response_cleaned = re.sub(r"```(?:python)?", "", response.strip())
                response_cleaned = re.sub(r"```", "", response_cleaned).strip()

                # Extract tuples using regex
                matches = re.findall(r'\(\s*[\'"](.+?)[\'"]\s*,\s*[\'"](.+?)[\'"]\s*,\s*[\'"](.+?)[\'"]\s*\)', response_cleaned)
                triples = [tuple(m) for m in matches]
            except Exception as e:
                print(f"[ERROR] Failed to parse triples from: {response}\nError: {e}")
                triples = []

            for triple in triples:
                all_triples.append(triple)
                triple_map[str(triple)] = idx

        return all_triples, triple_map
