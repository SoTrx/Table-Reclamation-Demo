import pytest

from table_reclamation.facade.table_reclamation import AccessPlanner

QUESTIONS = [
    "How can I solve a linear system 4x4?",
    "What are the basic concepts of learning linear optimization?",
    "How do I begin my study about linear optimization?",
    "What is the role of the objective function in finding the optimum solution in a linear optimization problem?",
    "What is the role of constraints in a linear optimization problem?",
    "How is the optimal solution usually found in a linear optimization problem?",
    "Fetch the assessments of student number 1273 and student number 3409",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_generate_mathe_plan(planner_mathe_split: AccessPlanner, question: str):
    planner_mathe_split.generate_stats()
    plan = planner_mathe_split.generate_plan(question)
    print(plan)
    assert len(plan) > 1
