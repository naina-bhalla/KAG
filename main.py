from src.builder.kg_builder import KnowledgeGraphBuilder
from src.solver.json_planner import JSONPlanner
from src.solver.executor import Executor
from src.model.kag_model import KagModel
from src.llm.cohere import Cohere
import os
from dotenv import load_dotenv

load_dotenv()  

api_key = os.getenv("COHERE_API_KEY")

from src.builder.kg_builder import KnowledgeGraphBuilder
from src.llm.cohere import Cohere
import os
from dotenv import load_dotenv

load_dotenv()
llm = Cohere(api_key=os.getenv("COHERE_API_KEY"))

docs = ["The doctor prescribed Ibuprofen to the patient suffering from arthritis."]
query = "What medication is used for arthritis?"
builder = KnowledgeGraphBuilder(llm=llm)
triples, triple_map = builder.build_(docs)

print("Triples:", triples)
print("Triple Map:", triple_map)

planner = JSONPlanner(llm)
plan = planner.plan(query)

executor = Executor(triples, triple_map)
context = executor.execute(plan)

model = KagModel(llm)
print("[DEBUG] Context given to LLM:", context)

answer = model.generate_answer(context)

print("Answer:", answer)