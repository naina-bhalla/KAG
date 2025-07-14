from difflib import SequenceMatcher

class Executor:
    def __init__(self, triples, triple_map):
        self.triples = triples
        self.triple_map = triple_map

    def similar(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def execute(self, plan: list):
        context = []
        for step in plan:
            if step['step'] == 'kg_retrieve':
                args = step['args']
                ent = args.get('entity')
                rel = args.get('relation')
                query = args.get('query')

                if query:
                    # Match query against stringified triples
                    for triple in self.triples:
                        joined = " ".join(triple).lower()
                        if any(word in joined for word in query.lower().split()):
                            context.append(triple)
                        elif self.similar(query, joined) > 0.5:
                            context.append(triple)

                elif ent:  # fall back to existing behavior
                    facts = [t for t in self.triples if t[0] == ent and (rel is None or t[1] == rel)]
                    context.extend(facts)

            elif step['step'] == 'combine':
                context = list(set(context))  # Deduplicate
        return context
