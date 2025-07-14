import cohere

class Cohere:
    def __init__(self, api_key, model="command-r"):
        self.client = cohere.Client(api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self.client.chat(
        model="command-r",
        message=prompt,
        temperature=0.3,
    )
        return response.text.strip()
