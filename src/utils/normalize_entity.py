from sentence_transformers import SentenceTransformer, util
import torch

class EntityNormalizer:
    def __init__(self, canonical_terms: list[str], model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.canonical_terms = canonical_terms
        self.term_embeddings = self.model.encode(canonical_terms, convert_to_tensor=True)

    def normalize(self, entity: str, threshold: float = 0.8) -> str:
        entity_embedding = self.model.encode(entity, convert_to_tensor=True)
        cosine_scores = util.cos_sim(entity_embedding, self.term_embeddings)[0]

        best_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()

        if best_score >= threshold:
            return self.canonical_terms[best_idx]
        else:
            return entity  # return as-is if not confident

