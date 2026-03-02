from table_reclamation.facade.table_reclamation import AccessPlanner


def test_generate_mathe_plan(planner_mathe: AccessPlanner):
    planner_mathe.generate_stats()
    plan = planner_mathe.generate_plan(
        "Discrete Mathematics Recursivity level 2")
    print(plan)
    assert len(plan) > 0
