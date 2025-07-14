import json
import re

class JSONPlanner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, query: str):
        prompt = f"""You are a helpful AI planner.
Given the user query: "{query}", respond with a JSON plan.
Do NOT add explanations or markdown. Just return JSON like this:

[
  {{
    "step": "kg_retrieve",
    "args": {{
      "query": "arthritis medication"
    }}
  }}
]
"""

        response = self.llm(prompt)
        print("[DEBUG] Raw LLM Response:\n", response)

        # ✅ Remove ```json ... ``` if it exists
        response_clean = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", response).strip()

        try:
            return json.loads(response_clean)
        except json.JSONDecodeError as e:
            print("❌ JSON decode error")
            print("Raw response:", response_clean)
            raise e
