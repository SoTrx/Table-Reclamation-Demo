from table_reclamation.facade.table_reclamation import AccessPlanner


def test_generate_mathe_plan(planner_mathe: AccessPlanner):
    planner_mathe.generate_stats()
    ap = planner_mathe.generate_plan(
        "Discrete Mathematics level 2 with Equations")
    print(ap)
