class KAGSolver:
    def __init__(self, llm, triples, triple_map):
        from solver.json_planner import JSONPlanner
        from solver.executor import Executor

        self.planner = JSONPlanner(llm)
        self.executor = Executor(triples, triple_map)

    def solve(self, question: str) -> list:
        plan = self.planner.plan(question)
        print("Execution plan:", plan)  # optional debug
        context = self.executor.execute(plan)
        return context
